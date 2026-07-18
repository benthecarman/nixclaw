"""Render the immutable OpenShell policy with concrete Nix store paths."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

app = typer.Typer(add_completion=False)

PLACEHOLDERS = (
    "NIXCLAW_AGENT_BIN",
    "NIXCLAW_PYTHON_BIN",
    "HERMES_BIN",
    "HERMES_ROOT",
    "NIXCLAW_ROOT",
    "BROKER_HOST",
    "BROKER_PORT",
    "INFERENCE_HOST",
    "INFERENCE_PORT",
)


def render_policy(template: str, values: dict[str, str]) -> str:
    missing = [name for name in PLACEHOLDERS if not values.get(name)]
    if missing:
        raise ValueError(f"Missing policy values: {', '.join(missing)}")
    rendered = template
    for name in PLACEHOLDERS:
        rendered = rendered.replace(f"@{name}@", values[name])
    unresolved = [name for name in PLACEHOLDERS if f"@{name}@" in rendered]
    if unresolved:
        raise ValueError(f"Unresolved policy values: {', '.join(unresolved)}")
    document = yaml.safe_load(rendered)
    if document.get("version") != 1:
        raise ValueError("Rendered policy must use OpenShell schema version 1")
    return rendered


@app.command()
def render(
    template: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    nixclaw_agent_bin: Annotated[str, typer.Option()],
    nixclaw_python_bin: Annotated[str, typer.Option()],
    hermes_bin: Annotated[str, typer.Option()],
    hermes_root: Annotated[str, typer.Option()],
    nixclaw_root: Annotated[str, typer.Option()],
    broker_host: Annotated[str, typer.Option()],
    broker_port: Annotated[int, typer.Option()],
    inference_host: Annotated[str, typer.Option()],
    inference_port: Annotated[int, typer.Option()],
) -> None:
    """Render and minimally validate a concrete OpenShell policy."""

    values = {
        "NIXCLAW_AGENT_BIN": nixclaw_agent_bin,
        "NIXCLAW_PYTHON_BIN": nixclaw_python_bin,
        "HERMES_BIN": hermes_bin,
        "HERMES_ROOT": hermes_root,
        "NIXCLAW_ROOT": nixclaw_root,
        "BROKER_HOST": broker_host,
        "BROKER_PORT": str(broker_port),
        "INFERENCE_HOST": inference_host,
        "INFERENCE_PORT": str(inference_port),
    }
    rendered = render_policy(template.read_text(), values)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    typer.echo(output)


if __name__ == "__main__":
    app()
