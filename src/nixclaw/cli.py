"""Hermes-facing NixClaw command-line interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID

import typer

from .broker import BrokerClient, BrokerError
from .knowledge import KnowledgeStore, environment_document
from .optimizer import NoCandidateError, Optimizer

app = typer.Typer(add_completion=False, no_args_is_help=True)
experiments_app = typer.Typer(add_completion=False, no_args_is_help=True)
lessons_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(experiments_app, name="experiments")
app.add_typer(lessons_app, name="lessons")


def _client() -> BrokerClient:
    base_url = os.environ.get("NIXCLAW_BROKER_URL")
    if not base_url:
        raise typer.BadParameter("NIXCLAW_BROKER_URL must be set")
    credential = os.environ.get(
        "NIXCLAW_BROKER_CREDENTIAL",
        "openshell:resolve:env:NIXCLAW_BROKER_TOKEN",
    )
    return BrokerClient(base_url, credential)


def _emit(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True)  # type: ignore[union-attr]
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _database_path() -> Path:
    state_directory = Path(os.environ.get("NIXCLAW_STATE_DIR", ".nixclaw"))
    return state_directory / "knowledge.sqlite3"


@app.command()
def facts() -> None:
    """Print sanitized environment facts."""

    try:
        with _client() as client:
            _emit(client.facts())
    except BrokerError as exc:
        _emit({"error": exc.code, "message": exc.message, "details": exc.details})
        raise typer.Exit(1) from exc


@app.command()
def config() -> None:
    """Print active configuration and tunable bounds."""

    try:
        with _client() as client:
            _emit(client.config())
    except BrokerError as exc:
        _emit({"error": exc.code, "message": exc.message, "details": exc.details})
        raise typer.Exit(1) from exc


@app.command()
def optimize(workload: str = "agent-tool") -> None:
    """Submit the best untried bounded experiment for a workload."""

    try:
        with _client() as client, KnowledgeStore(_database_path()) as store:
            experiment, candidate = Optimizer(client, store).propose(workload)
            _emit(
                {
                    "experiment": experiment.model_dump(mode="json", by_alias=True),
                    "candidateSource": candidate.source,
                    "nextAction": f"Wait for operator approval, then sync {experiment.id}",
                }
            )
    except (BrokerError, NoCandidateError, ValueError) as exc:
        code = exc.code if isinstance(exc, BrokerError) else "no_candidate"
        _emit({"error": code, "message": str(exc)})
        raise typer.Exit(1) from exc


@experiments_app.command("list")
def experiments_list() -> None:
    """List broker experiments."""

    try:
        with _client() as client:
            _emit([item.model_dump(mode="json", by_alias=True) for item in client.experiments()])
    except BrokerError as exc:
        _emit({"error": exc.code, "message": exc.message, "details": exc.details})
        raise typer.Exit(1) from exc


@experiments_app.command("show")
def experiments_show(experiment_id: UUID) -> None:
    """Show one broker experiment."""

    try:
        with _client() as client:
            _emit(client.experiment(experiment_id))
    except BrokerError as exc:
        _emit({"error": exc.code, "message": exc.message, "details": exc.details})
        raise typer.Exit(1) from exc


@experiments_app.command("sync")
def experiments_sync(experiment_id: UUID) -> None:
    """Ingest the current broker state and promote terminal evidence."""

    try:
        with _client() as client, KnowledgeStore(_database_path()) as store:
            _emit(Optimizer(client, store).sync(experiment_id))
    except (BrokerError, KeyError) as exc:
        code = exc.code if isinstance(exc, BrokerError) else "unknown_local_experiment"
        _emit({"error": code, "message": str(exc)})
        raise typer.Exit(1) from exc


@lessons_app.command("search")
def lessons_search(workload: str = "agent-tool", include_rejected: bool = False) -> None:
    """Retrieve compatible validated lessons for the current environment."""

    try:
        with _client() as client, KnowledgeStore(_database_path()) as store:
            facts = client.facts()
            config_value = client.config()
            environment = environment_document(facts, config_value, workload)
            lessons = store.search_lessons(
                environment,
                workload,
                include_rejected=include_rejected,
            )
            _emit(
                [
                    {
                        "id": str(lesson.id),
                        "compatibility": lesson.compatibility,
                        "symptom": lesson.symptom,
                        "repair": lesson.repair,
                        "evidence": lesson.evidence_summary,
                        "confidence": lesson.confidence,
                        "status": lesson.status,
                    }
                    for lesson in lessons
                ]
            )
    except BrokerError as exc:
        _emit({"error": exc.code, "message": exc.message, "details": exc.details})
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
