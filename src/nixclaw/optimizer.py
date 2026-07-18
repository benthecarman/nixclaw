"""Bounded, evidence-aware vLLM experiment selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from .broker import BrokerClient
from .knowledge import KnowledgeStore, canonical_json, environment_document
from .models import (
    Config,
    CreateExperimentRequest,
    Experiment,
    Facts,
    VllmProfilePatch,
)


class NoCandidateError(RuntimeError):
    """Raised when every legal candidate is already active or previously tried."""


@dataclass(frozen=True)
class Candidate:
    patch: VllmProfilePatch
    hypothesis: str
    source: str


class CandidateGenerator:
    """Generate a short ordered list from broker-advertised tunables."""

    def generate(
        self,
        facts: Facts,
        config: Config,
        workload_id: str,
        store: KnowledgeStore,
    ) -> list[Candidate]:
        environment = environment_document(facts, config, workload_id)
        attempted = store.attempted_patches(environment, workload_id)
        proposed: list[Candidate] = []

        for lesson in store.search_lessons(environment, workload_id):
            if self._legal_patch(lesson.repair, config) and self._changes_active(
                lesson.repair,
                config,
            ):
                proposed.append(
                    Candidate(
                        patch=VllmProfilePatch.model_validate(lesson.repair),
                        hypothesis=f"Revalidate prior lesson: {lesson.evidence_summary}",
                        source=f"lesson:{lesson.id}:{lesson.compatibility}",
                    )
                )

        active = config.active_profile
        tunables = config.tunable_fields
        if (
            workload_id == "agent-tools"
            and "enablePrefixCaching" in tunables
            and not active.get("enablePrefixCaching", False)
        ):
            proposed.append(
                Candidate(
                    patch=VllmProfilePatch(enable_prefix_caching=True),
                    hypothesis="Repeated prefixes should improve throughput with prefix caching.",
                    source="heuristic:prefix-cache",
                )
            )

        if "maxNumSeqs" in tunables:
            field = tunables["maxNumSeqs"]
            target = max(4 if workload_id == "agent-tools" else 1, int(field.minimum or 1))
            if field.maximum is not None:
                target = min(target, int(field.maximum))
            if active.get("maxNumSeqs") != target:
                proposed.append(
                    Candidate(
                        patch=VllmProfilePatch(max_num_seqs=target),
                        hypothesis=f"Match sequence capacity to the {workload_id} concurrency.",
                        source="heuristic:sequence-capacity",
                    )
                )

        peak_memory = store.latest_peak_memory_ratio(environment)
        if peak_memory is not None and peak_memory < 0.85 and "gpuMemoryUtilization" in tunables:
            field = tunables["gpuMemoryUtilization"]
            current = float(active.get("gpuMemoryUtilization", 0.76))
            step = float(field.step or 0.01)
            maximum = float(field.maximum or 0.95)
            target = min(maximum, round(current + step, 4))
            if target > current:
                proposed.append(
                    Candidate(
                        patch=VllmProfilePatch(gpu_memory_utilization=target),
                        hypothesis="Unused GPU memory may support a larger cache without pressure.",
                        source="heuristic:memory-utilization",
                    )
                )

        if "enforceEager" in tunables and active.get("enforceEager", False):
            proposed.append(
                Candidate(
                    patch=VllmProfilePatch(enforce_eager=False),
                    hypothesis="CUDA graphs may improve steady-state inference performance.",
                    source="heuristic:cuda-graphs",
                )
            )

        unique: list[Candidate] = []
        seen: set[str] = set()
        for candidate in proposed:
            encoded = canonical_json(candidate.patch.supplied())
            if encoded in seen or encoded in attempted:
                continue
            seen.add(encoded)
            unique.append(candidate)
        return unique[:3]

    @staticmethod
    def _legal_patch(patch: dict[str, Any], config: Config) -> bool:
        return bool(patch) and set(patch).issubset(config.tunable_fields)

    @staticmethod
    def _changes_active(patch: dict[str, Any], config: Config) -> bool:
        return any(config.active_profile.get(key) != value for key, value in patch.items())


class Optimizer:
    """Submit and resume one bounded experiment at a time."""

    def __init__(self, broker: BrokerClient, store: KnowledgeStore) -> None:
        self.broker = broker
        self.store = store
        self.generator = CandidateGenerator()

    def propose(self, workload_id: str) -> tuple[Experiment, Candidate]:
        facts = self.broker.facts()
        config = self.broker.config()
        if workload_id not in config.workload_ids:
            raise ValueError(f"Broker does not advertise workload {workload_id!r}")
        environment = environment_document(facts, config, workload_id)
        candidates = self.generator.generate(facts, config, workload_id, self.store)
        if not candidates:
            raise NoCandidateError("No untried legal profile candidates remain")
        candidate = candidates[0]
        episode_id = self.store.start_episode(
            request=f"Optimize workload {workload_id}",
            environment=environment,
            model=facts.served_model,
        )
        request = CreateExperimentRequest(
            base_generation=config.base_generation,
            workload_id=workload_id,
            hypothesis=candidate.hypothesis,
            profile_patch=candidate.patch,
            client_request_id=uuid4(),
        )
        try:
            experiment = self.broker.create_experiment(request)
        except Exception as exc:
            self.store.finish_episode(episode_id, "failed", str(exc))
            raise
        self.store.record_submission(episode_id, experiment, environment)
        self.store.finish_episode(episode_id, "submitted", str(experiment.id))
        return experiment, candidate

    def sync(self, experiment_id: UUID) -> Experiment:
        experiment = self.broker.experiment(experiment_id)
        self.store.ingest_experiment(experiment)
        return experiment
