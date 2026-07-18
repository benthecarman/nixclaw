"""Deterministic fixture implementation of the broker v1 contract."""

from __future__ import annotations

from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

import httpx
import typer
from pydantic import ValidationError

from .models import CreateExperimentRequest, Experiment, ExperimentState

app = typer.Typer(add_completion=False, help="Run a deterministic NixClaw broker fixture.")


def _envelope(data: Any, request_id: UUID | None = None) -> dict[str, Any]:
    return {
        "schemaVersion": "1",
        "requestId": str(request_id or uuid4()),
        "data": data,
    }


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schemaVersion": "1",
        "requestId": str(uuid4()),
        "error": {"code": code, "message": message, "details": details or {}},
    }


class FixtureBroker:
    """Stateful request handler usable by tests and the local fixture server."""

    generation = "/nix/store/fixture-nixos-system"
    profile_hash = "sha256:baseline"

    def __init__(self) -> None:
        self._experiments: dict[UUID, Experiment] = {}
        self._idempotency: dict[UUID, UUID] = {}
        self._lock = Lock()

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/v1/facts":
            return self._response(200, _envelope(self._facts()))
        if request.method == "GET" and path == "/v1/config":
            return self._response(200, _envelope(self._config()))
        if request.method == "GET" and path == "/v1/experiments":
            values = [self._dump(item) for item in self._experiments.values()]
            return self._response(200, _envelope(values))
        if request.method == "GET" and path.startswith("/v1/experiments/"):
            try:
                experiment_id = UUID(path.rsplit("/", 1)[-1])
                experiment = self._experiments[experiment_id]
            except (ValueError, KeyError):
                return self._response(404, _error("not_found", "Experiment was not found"))
            return self._response(200, _envelope(self._dump(experiment)))
        if request.method == "POST" and path == "/v1/experiments":
            return self._create_experiment(request)
        if request.method == "POST" and path == "/v1/proposals":
            return self._response(
                202,
                _envelope({"id": str(uuid4()), "state": "awaitingApproval"}),
            )
        return self._response(404, _error("route_not_found", "Route is not exposed"))

    def advance(self, experiment_id: UUID, state: ExperimentState) -> Experiment:
        with self._lock:
            experiment = self._experiments[experiment_id]
            experiment.state = state
            experiment.updated_at = datetime.now(UTC)
            if state == ExperimentState.ACCEPTED:
                experiment.candidate_profile_hash = "sha256:candidate"
                experiment.candidate_generation = "/nix/store/fixture-candidate-system"
                experiment.decision = {"accepted": True, "gates": []}
            return experiment

    def _create_experiment(self, request: httpx.Request) -> httpx.Response:
        try:
            parsed = CreateExperimentRequest.model_validate_json(request.content)
        except ValidationError as exc:
            return self._response(
                422,
                _error(
                    "invalid_request",
                    "Request failed schema validation",
                    {"errors": exc.errors(include_url=False)},
                ),
            )
        if parsed.base_generation != self.generation:
            return self._response(
                409,
                _error(
                    "stale_generation",
                    "Base generation is no longer active",
                    {"currentGeneration": self.generation},
                ),
            )
        supplied = parsed.profile_patch.supplied()
        allowed = self._config()["tunables"]
        unknown = sorted(set(supplied) - set(allowed))
        if unknown:
            return self._response(
                422,
                _error(
                    "unsupported_tunable",
                    "Patch contains unsupported fields",
                    {"fields": unknown},
                ),
            )
        with self._lock:
            if parsed.client_request_id in self._idempotency:
                existing_id = self._idempotency[parsed.client_request_id]
                return self._response(200, _envelope(self._dump(self._experiments[existing_id])))
            now = datetime.now(UTC)
            experiment = Experiment(
                id=uuid4(),
                state=ExperimentState.AWAITING_APPROVAL,
                base_generation=parsed.base_generation,
                workload_id=parsed.workload_id,
                hypothesis=parsed.hypothesis,
                profile_patch=parsed.profile_patch,
                client_request_id=parsed.client_request_id,
                original_profile_hash=self.profile_hash,
                created_at=now,
                updated_at=now,
            )
            self._experiments[experiment.id] = experiment
            self._idempotency[parsed.client_request_id] = experiment.id
        return self._response(202, _envelope(self._dump(experiment)))

    @staticmethod
    def _dump(model: Experiment) -> dict[str, Any]:
        return model.model_dump(mode="json", by_alias=True)

    @staticmethod
    def _response(status: int, payload: dict[str, Any]) -> httpx.Response:
        return httpx.Response(status, json=payload)

    def _facts(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "nixRevision": "fixture-revision",
            "architecture": "aarch64-linux",
            "gpus": [
                {
                    "index": 0,
                    "model": "NVIDIA GB10",
                    "computeCapability": "12.1",
                    "memoryBytes": 128_849_018_880,
                }
            ],
            "resources": {
                "cpuCount": 20,
                "memoryBytes": 129_922_760_704,
                "swapBytes": 0,
            },
            "cluster": [{"nodeId": "nixos-s6", "role": "head", "rank": 0, "healthy": True}],
            "vllm": {
                "version": "0.25.1",
                "model": "fixture/model",
                "profileHash": self.profile_hash,
                "healthy": True,
            },
        }

    def _config(self) -> dict[str, Any]:
        return {
            "baseGeneration": self.generation,
            "activeProfileName": "baseline",
            "profileHash": self.profile_hash,
            "servedModel": "fixture/model",
            "activeProfile": {
                "gpuMemoryUtilization": 0.76,
                "maxModelLen": 65536,
                "enablePrefixCaching": False,
                "enforceEager": False,
            },
            "workloadIds": ["interactive", "agent-tool"],
            "tunables": {
                "gpuMemoryUtilization": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 0.95,
                    "step": 0.05,
                },
                "maxNumSeqs": {"type": "integer", "minimum": 1, "maximum": 32, "step": 1},
                "enablePrefixCaching": {"type": "boolean"},
                "enableChunkedPrefill": {"type": "boolean", "nullable": True},
                "enforceEager": {"type": "boolean"},
            },
        }


class _Handler(BaseHTTPRequestHandler):
    broker: FixtureBroker

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        content = self.rfile.read(length)
        request = httpx.Request(
            self.command,
            f"http://fixture{urlparse(self.path).path}",
            headers=dict(self.headers),
            content=content,
        )
        response = self.broker.handle(request)
        self.send_response(response.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(response.content)

    def log_message(self, format: str, *args: object) -> None:
        typer.echo(format % args)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the fixture until interrupted."""

    _Handler.broker = FixtureBroker()
    server = ThreadingHTTPServer((host, port), _Handler)
    typer.echo(f"NixClaw fixture listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    app()
