"""Live containment checks intended to run inside the OpenShell sandbox."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from .broker import BrokerClient, BrokerError

app = typer.Typer(add_completion=False)


def run_checks(
    broker_url: str,
    credential: str,
    external_url: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    with BrokerClient(broker_url, credential, transport=transport) as broker:
        try:
            broker.facts()
            checks.append({"name": "allowed_facts", "passed": True})
        except BrokerError as exc:
            checks.append({"name": "allowed_facts", "passed": False, "detail": str(exc)})

    headers = {"Authorization": f"Bearer {credential}"}
    with httpx.Client(
        follow_redirects=False,
        timeout=httpx.Timeout(5, connect=2),
        transport=transport,
    ) as client:
        for name, url in (
            ("blocked_activator_route", broker_url.rstrip("/") + "/v1/approve"),
            ("blocked_external_egress", external_url),
        ):
            try:
                response = client.get(url, headers=headers)
                checks.append(
                    {
                        "name": name,
                        "passed": response.status_code in {401, 403, 404, 405},
                        "detail": f"HTTP {response.status_code}",
                    }
                )
            except httpx.HTTPError as exc:
                checks.append({"name": name, "passed": True, "detail": str(exc)})

    for path in (
        Path("/run/nixclaw/activator.sock"),
        Path("/nix/var/nix/daemon-socket/socket"),
        Path("/run/docker.sock"),
        Path("/etc/nixos"),
    ):
        try:
            if path.is_dir():
                list(path.iterdir())
            else:
                with path.open("rb") as protected_file:
                    protected_file.read(1)
            passed = False
            detail = "protected path was readable"
        except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as exc:
            passed = True
            detail = type(exc).__name__
        checks.append({"name": f"blocked_path:{path}", "passed": passed, "detail": detail})
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


@app.command()
def check(
    broker_url: Annotated[str | None, typer.Option()] = None,
    external_url: Annotated[str, typer.Option()] = "https://example.com/",
) -> None:
    """Verify allowed broker reads and expected containment denials."""

    resolved_broker = broker_url or os.environ.get("NIXCLAW_BROKER_URL")
    if not resolved_broker:
        raise typer.BadParameter("NIXCLAW_BROKER_URL or --broker-url is required")
    credential = os.environ.get(
        "NIXCLAW_BROKER_CREDENTIAL",
        "openshell:resolve:env:NIXCLAW_BROKER_TOKEN",
    )
    result = run_checks(resolved_broker, credential, external_url)
    typer.echo(__import__("json").dumps(result, indent=2))
    if not result["passed"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
