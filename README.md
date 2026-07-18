# NixClaw agent software

NixClaw is the untrusted-side learning and measurement component described in
[`HACKATHON_PLAN.md`](HACKATHON_PLAN.md). It gives Hermes a constrained broker
client, a persistent evidence store, a bounded vLLM optimizer, correctness and
performance benchmarks, and a read-only scorecard.

The privileged NixOS broker and activator live in
[`benthecarman/nemoclaw-nix`](https://github.com/benthecarman/nemoclaw-nix).
The versioned schemas in `schemas/nixclaw/v1` are the integration contract
between the two repositories.

## Development

```console
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Start the deterministic broker fixture with:

```console
uv run nixclaw-fixture --port 8765
```

Point the client at it and exercise the recursive loop:

```console
export NIXCLAW_BROKER_URL=http://127.0.0.1:8765
uv run nixclaw-agent facts
uv run nixclaw-agent optimize --workload agent-tools
```

The optimizer stops at host approval. Once an experiment reaches a terminal
state, synchronize it with `nixclaw-agent experiments sync ID` so accepted or
negative evidence becomes available to future sessions.

Generate the read-only scorecard with:

```console
uv run nixclaw-scorecard --output-directory scorecard-output
```

See [`docs/integration.md`](docs/integration.md) for broker, activator,
benchmark, OpenShell, and DGX Spark integration details.
