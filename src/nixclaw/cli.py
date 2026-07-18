"""Hermes-facing NixClaw command-line interface."""

from __future__ import annotations

import json
import os
from uuid import UUID

import typer

from .broker import BrokerClient, BrokerError

app = typer.Typer(add_completion=False, no_args_is_help=True)
experiments_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(experiments_app, name="experiments")


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


if __name__ == "__main__":
    app()
