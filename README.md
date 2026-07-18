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
