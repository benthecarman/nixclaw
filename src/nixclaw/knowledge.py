"""Persistent, auditable evidence storage for NixClaw experiments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from .models import Config, Experiment, ExperimentState, Facts


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def environment_document(facts: Facts, config: Config, workload_id: str) -> dict[str, Any]:
    """Return the stable compatibility fields used to scope lessons."""

    return {
        "architecture": facts.architecture,
        "gpus": [
            {
                "model": gpu.model,
                "computeCapability": gpu.compute_capability,
                "memoryBytes": gpu.memory_bytes,
            }
            for gpu in facts.gpus
        ],
        "nixRevision": facts.nix_revision,
        "vllmVersion": facts.vllm.version,
        "servedModel": config.served_model,
        "cluster": [
            {"role": node.role, "rank": node.rank}
            for node in sorted(facts.cluster, key=lambda item: item.rank)
        ],
        "workloadId": workload_id,
    }


def environment_fingerprint(document: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(document).encode()).hexdigest()


@dataclass(frozen=True)
class Lesson:
    id: UUID
    symptom: str
    workload_id: str
    environment_fingerprint: str
    environment: dict[str, Any]
    cause: str
    repair: dict[str, Any]
    evidence_summary: str
    confidence: float
    status: str
    compatibility: str = "exact"


class KnowledgeStore(AbstractContextManager["KnowledgeStore"]):
    """SQLite store whose contents are evidence, never activation authority."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __exit__(self, *_args: object) -> None:
        self.close()

    def start_episode(
        self,
        request: str,
        environment: dict[str, Any],
        model: str,
    ) -> UUID:
        episode_id = uuid4()
        fingerprint = environment_fingerprint(environment)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO episodes (
                    id, request, environment_fingerprint, environment_json,
                    model, status, started_at
                ) VALUES (?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    str(episode_id),
                    request,
                    fingerprint,
                    canonical_json(environment),
                    model,
                    utc_now(),
                ),
            )
        return episode_id

    def finish_episode(self, episode_id: UUID, status: str, result: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE episodes
                SET status = ?, result = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, result, utc_now(), str(episode_id)),
            )

    def record_submission(
        self,
        episode_id: UUID,
        experiment: Experiment,
        environment: dict[str, Any],
    ) -> UUID:
        change_id = uuid4()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO changes (
                    id, episode_id, experiment_id, environment_fingerprint,
                    environment_json, workload_id, domain, profile_patch_json,
                    base_generation, candidate_generation, outcome, created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'inference', ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(change_id),
                    str(episode_id),
                    str(experiment.id),
                    environment_fingerprint(environment),
                    canonical_json(environment),
                    experiment.workload_id,
                    canonical_json(experiment.profile_patch.supplied()),
                    experiment.base_generation,
                    experiment.candidate_generation,
                    experiment.state.value,
                    utc_now(),
                    utc_now(),
                ),
            )
        return change_id

    def ingest_experiment(self, experiment: Experiment) -> None:
        row = self.connection.execute(
            "SELECT id, episode_id FROM changes WHERE experiment_id = ?",
            (str(experiment.id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Experiment {experiment.id} has no local submission")
        change_id = row["id"]
        with self.connection:
            self.connection.execute(
                """
                UPDATE changes
                SET candidate_generation = ?, outcome = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    experiment.candidate_generation,
                    experiment.state.value,
                    utc_now(),
                    change_id,
                ),
            )
            for kind, payload in (
                ("baseline", experiment.baseline_result),
                ("candidate", experiment.candidate_result),
                ("decision", experiment.decision),
            ):
                if payload is not None:
                    self.connection.execute(
                        """
                        INSERT INTO metrics (id, change_id, kind, payload_json, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(change_id, kind) DO UPDATE SET
                            payload_json = excluded.payload_json,
                            created_at = excluded.created_at
                        """,
                        (str(uuid4()), change_id, kind, canonical_json(payload), utc_now()),
                    )

        if experiment.state == ExperimentState.ACCEPTED:
            self._promote_lesson(change_id, experiment)
        elif experiment.state in {
            ExperimentState.REJECTED,
            ExperimentState.ROLLED_BACK,
            ExperimentState.FAILED,
        }:
            self._record_negative_lesson(change_id, experiment)

    def attempted_patches(
        self,
        environment: dict[str, Any],
        workload_id: str,
    ) -> set[str]:
        rows = self.connection.execute(
            """
            SELECT profile_patch_json FROM changes
            WHERE environment_fingerprint = ? AND workload_id = ?
            """,
            (environment_fingerprint(environment), workload_id),
        ).fetchall()
        return {row["profile_patch_json"] for row in rows}

    def search_lessons(
        self,
        environment: dict[str, Any],
        workload_id: str,
        *,
        include_rejected: bool = False,
        limit: int = 10,
    ) -> list[Lesson]:
        statuses = ("validated", "rejected") if include_rejected else ("validated",)
        placeholders = ",".join("?" for _ in statuses)
        rows = self.connection.execute(
            f"""
            SELECT * FROM lessons
            WHERE workload_id = ? AND status IN ({placeholders})
            ORDER BY confidence DESC, updated_at DESC
            """,  # noqa: S608 - placeholders are generated, not user supplied
            (workload_id, *statuses),
        ).fetchall()
        target_fingerprint = environment_fingerprint(environment)
        ranked: list[tuple[int, Lesson]] = []
        for row in rows:
            saved_environment = json.loads(row["environment_json"])
            compatibility = self._compatibility(environment, saved_environment)
            if compatibility is None:
                continue
            lesson = Lesson(
                id=UUID(row["id"]),
                symptom=row["symptom"],
                workload_id=row["workload_id"],
                environment_fingerprint=row["environment_fingerprint"],
                environment=saved_environment,
                cause=row["cause"],
                repair=json.loads(row["repair_json"]),
                evidence_summary=row["evidence_summary"],
                confidence=row["confidence"],
                status=row["status"],
                compatibility=(
                    "exact" if row["environment_fingerprint"] == target_fingerprint else "transfer"
                ),
            )
            ranked.append((2 if lesson.compatibility == "exact" else 1, lesson))
        ranked.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
        return [item[1] for item in ranked[:limit]]

    def latest_peak_memory_ratio(self, environment: dict[str, Any]) -> float | None:
        row = self.connection.execute(
            """
            SELECT m.payload_json
            FROM metrics AS m
            JOIN changes AS c ON c.id = m.change_id
            WHERE c.environment_fingerprint = ? AND m.kind = 'candidate'
            ORDER BY m.created_at DESC LIMIT 1
            """,
            (environment_fingerprint(environment),),
        ).fetchone()
        if row is None:
            return None
        value = json.loads(row["payload_json"]).get("summary", {}).get("peakMemoryRatio")
        return float(value) if value is not None else None

    def list_experiments(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT experiment_id, workload_id, profile_patch_json, outcome,
                   base_generation, candidate_generation, created_at, updated_at
            FROM changes ORDER BY created_at DESC
            """
        ).fetchall()
        return [
            {
                **dict(row),
                "profile_patch": json.loads(row["profile_patch_json"]),
            }
            for row in rows
        ]

    def _promote_lesson(self, change_id: str, experiment: Experiment) -> None:
        row = self.connection.execute(
            "SELECT * FROM changes WHERE id = ?",
            (change_id,),
        ).fetchone()
        assert row is not None
        decision = experiment.decision or {}
        deltas = decision.get("deltas", {})
        throughput_delta = deltas.get("throughputPercent")
        evidence = "Experiment passed every acceptance gate."
        if throughput_delta is not None:
            evidence = f"Throughput changed by {float(throughput_delta):.2f}% and all gates passed."
        self._insert_lesson(
            change_id=change_id,
            symptom=experiment.hypothesis,
            workload_id=experiment.workload_id,
            environment_fingerprint_value=row["environment_fingerprint"],
            environment_json=row["environment_json"],
            cause="The previous serving profile was suboptimal for this workload.",
            repair=experiment.profile_patch.supplied(),
            evidence_summary=evidence,
            confidence=0.8,
            status="validated",
        )

    def _record_negative_lesson(self, change_id: str, experiment: Experiment) -> None:
        row = self.connection.execute(
            "SELECT * FROM changes WHERE id = ?",
            (change_id,),
        ).fetchone()
        assert row is not None
        reason = experiment.rollback_reason or f"Experiment ended as {experiment.state.value}."
        self._insert_lesson(
            change_id=change_id,
            symptom=experiment.hypothesis,
            workload_id=experiment.workload_id,
            environment_fingerprint_value=row["environment_fingerprint"],
            environment_json=row["environment_json"],
            cause="The proposed profile did not satisfy the acceptance policy.",
            repair=experiment.profile_patch.supplied(),
            evidence_summary=reason,
            confidence=0.9,
            status="rejected",
        )

    def _insert_lesson(
        self,
        *,
        change_id: str,
        symptom: str,
        workload_id: str,
        environment_fingerprint_value: str,
        environment_json: str,
        cause: str,
        repair: dict[str, Any],
        evidence_summary: str,
        confidence: float,
        status: str,
    ) -> None:
        existing = self.connection.execute(
            "SELECT 1 FROM lesson_evidence WHERE change_id = ?",
            (change_id,),
        ).fetchone()
        if existing is not None:
            return
        lesson_id = uuid4()
        now = utc_now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO lessons (
                    id, symptom, workload_id, environment_fingerprint,
                    environment_json, cause, repair_json, evidence_summary,
                    confidence, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(lesson_id),
                    symptom,
                    workload_id,
                    environment_fingerprint_value,
                    environment_json,
                    cause,
                    canonical_json(repair),
                    evidence_summary,
                    confidence,
                    status,
                    now,
                    now,
                ),
            )
            self.connection.execute(
                "INSERT INTO lesson_evidence (lesson_id, change_id) VALUES (?, ?)",
                (str(lesson_id), change_id),
            )
            self.connection.execute(
                """
                INSERT INTO edges (
                    id, source_type, source_id, relation, target_type,
                    target_id, evidence_change_id, created_at
                ) VALUES (?, 'lesson', ?, 'supported_by', 'change', ?, ?, ?)
                """,
                (str(uuid4()), str(lesson_id), change_id, change_id, now),
            )

    @staticmethod
    def _compatibility(target: dict[str, Any], saved: dict[str, Any]) -> str | None:
        if target == saved:
            return "exact"
        required_equal = ("architecture", "vllmVersion", "servedModel", "gpus", "cluster")
        if all(target.get(key) == saved.get(key) for key in required_equal):
            return "transfer"
        return None

    def _migrate(self) -> None:
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version > 1:
            raise RuntimeError(f"Knowledge database version {version} is newer than this binary")
        if version == 0:
            with self.connection:
                self.connection.executescript(
                    """
                    CREATE TABLE episodes (
                        id TEXT PRIMARY KEY,
                        request TEXT NOT NULL,
                        environment_fingerprint TEXT NOT NULL,
                        environment_json TEXT NOT NULL,
                        model TEXT NOT NULL,
                        result TEXT,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        completed_at TEXT
                    );
                    CREATE TABLE changes (
                        id TEXT PRIMARY KEY,
                        episode_id TEXT NOT NULL REFERENCES episodes(id),
                        experiment_id TEXT NOT NULL UNIQUE,
                        environment_fingerprint TEXT NOT NULL,
                        environment_json TEXT NOT NULL,
                        workload_id TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        profile_patch_json TEXT NOT NULL,
                        base_generation TEXT NOT NULL,
                        candidate_generation TEXT,
                        outcome TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX changes_environment_idx
                    ON changes(environment_fingerprint, workload_id);
                    CREATE TABLE metrics (
                        id TEXT PRIMARY KEY,
                        change_id TEXT NOT NULL REFERENCES changes(id),
                        kind TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE (change_id, kind)
                    );
                    CREATE TABLE lessons (
                        id TEXT PRIMARY KEY,
                        symptom TEXT NOT NULL,
                        workload_id TEXT NOT NULL,
                        environment_fingerprint TEXT NOT NULL,
                        environment_json TEXT NOT NULL,
                        cause TEXT NOT NULL,
                        repair_json TEXT NOT NULL,
                        evidence_summary TEXT NOT NULL,
                        confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                        status TEXT NOT NULL CHECK(
                            status IN (
                                'candidate', 'validated', 'rejected', 'superseded'
                            )
                        ),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX lessons_lookup_idx
                    ON lessons(workload_id, status, environment_fingerprint);
                    CREATE TABLE lesson_evidence (
                        lesson_id TEXT NOT NULL REFERENCES lessons(id),
                        change_id TEXT NOT NULL REFERENCES changes(id),
                        PRIMARY KEY (lesson_id, change_id)
                    );
                    CREATE TABLE edges (
                        id TEXT PRIMARY KEY,
                        source_type TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        target_type TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        evidence_change_id TEXT REFERENCES changes(id),
                        created_at TEXT NOT NULL
                    );
                    PRAGMA user_version = 1;
                    """
                )
