# ruff: noqa: E501
"""Generate a read-only, self-contained environment scorecard."""

from __future__ import annotations

import html
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer

from .broker import BrokerClient
from .knowledge import KnowledgeStore
from .models import Config, Facts

app = typer.Typer(add_completion=False)


def build_scorecard(facts: Facts, config: Config, store: KnowledgeStore) -> dict[str, Any]:
    evidence = store.scorecard_snapshot()
    return {
        "schemaVersion": "1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "environment": {
            "generation": facts.generation,
            "nixRevision": facts.nixos_revision,
            "architecture": facts.architecture,
            "gpus": [gpu.model for gpu in facts.gpu],
            "clusterNodes": len(facts.cluster_nodes),
            "vllmVersion": facts.vllm_version,
            "servedModel": config.served_model,
            "profileName": config.active_profile_name,
            "profileHash": config.active_profile_hash,
            "healthy": all(service.healthy for service in facts.services)
            and all(node.healthy for node in facts.cluster_nodes),
        },
        **evidence,
    }


def render_scorecard(scorecard: dict[str, Any]) -> str:
    environment = scorecard["environment"]
    episodes = scorecard["episodes"]
    experiments = scorecard["experiments"]
    lessons = scorecard["lessons"]
    def escape(value: object) -> str:
        return html.escape(str(value))

    def metric(result: dict[str, Any], name: str, statistic: str, suffix: str) -> str:
        value = result.get(name, {}).get(statistic)
        return "Unavailable" if value is None else f"{float(value):.2f}{suffix}"

    health_class = "good" if environment["healthy"] else "bad"
    latest_measured = next(
        (
            item
            for item in experiments
            if "baseline" in item["metrics"] and "candidate" in item["metrics"]
        ),
        None,
    )
    if latest_measured:
        baseline_summary = latest_measured["metrics"]["baseline"]
        candidate_summary = latest_measured["metrics"]["candidate"]
        decision = latest_measured["metrics"].get("decision", {})
    else:
        baseline_summary = {}
        candidate_summary = {}
        decision = {}
    metric_cards = "".join(
        f"<div class='card'>{escape(label)}"
        f"<span class='value'>{escape(metric(baseline_summary, name, statistic, suffix))} → "
        f"{escape(metric(candidate_summary, name, statistic, suffix))}</span></div>"
        for label, name, statistic, suffix in (
            ("Throughput", "outputTokensPerSecond", "median", " tok/s"),
            ("p95 TTFT", "ttftMs", "p95", " ms"),
            ("p95 inter-token", "interTokenLatencyMs", "p95", " ms"),
        )
    )
    gates = [
        (code, True) for code in decision.get("passedGates", [])
    ] + [
        (code, False) for code in decision.get("failedGates", [])
    ]
    explanations = decision.get("explanations", [])
    gate_rows = "".join(
        "<tr>"
        f"<td><code>{escape(code)}</code></td>"
        f"<td class='{'good' if passed else 'bad'}'>"
        f"{'pass' if passed else 'fail'}</td>"
        f"<td>{escape(explanations[index] if index < len(explanations) else '')}</td>"
        "</tr>"
        for index, (code, passed) in enumerate(gates)
    ) or "<tr><td colspan='3' class='empty'>No measured decision recorded</td></tr>"
    experiment_rows = "".join(
        "<tr>"
        f"<td><code>{escape(item['experimentId'])}</code></td>"
        f"<td>{escape(item['workloadId'])}</td>"
        f"<td><span class='pill {escape(item['outcome'])}'>{escape(item['outcome'])}</span></td>"
        f"<td><code>{escape(json.dumps(item['profilePatch'], sort_keys=True))}</code></td>"
        "</tr>"
        for item in experiments
    ) or "<tr><td colspan='4' class='empty'>No experiments recorded</td></tr>"
    lesson_rows = "".join(
        "<article class='lesson'>"
        f"<div><span class='pill {escape(item['status'])}'>{escape(item['status'])}</span> "
        f"<strong>{escape(item['workloadId'])}</strong></div>"
        f"<p>{escape(item['evidence'])}</p>"
        f"<code>{escape(json.dumps(item['repair'], sort_keys=True))}</code>"
        "</article>"
        for item in lessons
    ) or "<p class='empty'>No lessons recorded</p>"
    embedded = json.dumps(scorecard, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NixClaw scorecard</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #0b1020; color: #edf2f7; }}
main {{ max-width: 1120px; margin: auto; padding: 32px 20px 64px; }}
h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 36px; }}
.muted, .empty {{ color: #94a3b8; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.card, .lesson {{ background: #151d32; border: 1px solid #29334d; border-radius: 10px; padding: 16px; }}
.value {{ display: block; font-size: 1.35rem; margin-top: 6px; overflow-wrap: anywhere; }}
.good {{ color: #69db9c; }} .bad {{ color: #ff7b86; }}
table {{ width: 100%; border-collapse: collapse; background: #151d32; }}
th, td {{ padding: 12px; border-bottom: 1px solid #29334d; text-align: left; vertical-align: top; }}
code {{ color: #c4b5fd; overflow-wrap: anywhere; }}
.pill {{ display: inline-block; border-radius: 999px; padding: 2px 8px; background: #334155; font-size: .8rem; }}
.accepted, .validated {{ background: #14532d; }} .rejected, .rolledBack, .failed {{ background: #7f1d1d; }}
.lesson {{ margin-bottom: 10px; }}
</style>
</head>
<body><main>
<h1>NixClaw environment scorecard</h1>
<p class="muted">Generated {escape(scorecard['generatedAt'])}. Read-only; approval and activation are intentionally absent.</p>
<section class="grid">
  <div class="card">Health<span class="value {health_class}">{'Healthy' if environment['healthy'] else 'Unhealthy'}</span></div>
  <div class="card">Generation<span class="value"><code>{escape(environment['generation'])}</code></span></div>
  <div class="card">vLLM<span class="value">{escape(environment['vllmVersion'])}</span></div>
  <div class="card">Profile<span class="value">{escape(environment['profileName'])}</span></div>
  <div class="card">Episodes<span class="value">{escape(episodes['total'])}</span></div>
  <div class="card">Validated lessons<span class="value">{sum(item['status'] == 'validated' for item in lessons)}</span></div>
</section>
<h2>Environment</h2>
<section class="grid">
  <div class="card">Architecture<span class="value">{escape(environment['architecture'])}</span></div>
  <div class="card">GPU<span class="value">{escape(', '.join(environment['gpus']))}</span></div>
  <div class="card">Model<span class="value">{escape(environment['servedModel'])}</span></div>
  <div class="card">Cluster nodes<span class="value">{escape(environment['clusterNodes'])}</span></div>
</section>
<h2>Latest measurement</h2>
<p class="muted">Baseline → candidate</p>
<section class="grid">{metric_cards}</section>
<table><thead><tr><th>Gate</th><th>Result</th><th>Evidence</th></tr></thead>
<tbody>{gate_rows}</tbody></table>
<h2>Experiments</h2>
<table><thead><tr><th>ID</th><th>Workload</th><th>Outcome</th><th>Patch</th></tr></thead>
<tbody>{experiment_rows}</tbody></table>
<h2>Lessons</h2>
{lesson_rows}
<script id="nixclaw-scorecard" type="application/json">{embedded}</script>
</main></body></html>
"""


def write_scorecard(output_directory: Path, scorecard: dict[str, Any]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        output_directory / "scorecard.json": json.dumps(scorecard, indent=2, sort_keys=True),
        output_directory / "index.html": render_scorecard(scorecard),
    }
    for path, content in outputs.items():
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(content)
        temporary.replace(path)


@app.command()
def generate(
    output_directory: Annotated[Path, typer.Option()] = Path("scorecard-output"),
    database: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Read broker and evidence state and produce static scorecard files."""

    broker_url = os.environ.get("NIXCLAW_BROKER_URL")
    if not broker_url:
        raise typer.BadParameter("NIXCLAW_BROKER_URL must be set")
    database_path = database or Path(os.environ.get("NIXCLAW_STATE_DIR", ".nixclaw")) / "knowledge.sqlite3"
    credential = os.environ.get(
        "NIXCLAW_BROKER_CREDENTIAL",
        "openshell:resolve:env:NIXCLAW_BROKER_TOKEN",
    )
    with BrokerClient(broker_url, credential) as broker, KnowledgeStore(database_path) as store:
        scorecard = build_scorecard(broker.facts(), broker.config(), store)
    write_scorecard(output_directory, scorecard)
    typer.echo(output_directory / "index.html")


if __name__ == "__main__":
    app()
