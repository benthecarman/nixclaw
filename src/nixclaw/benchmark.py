"""Model-configurable vLLM benchmark and acceptance evaluator."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Annotated, Any
from uuid import uuid4

import httpx
import typer
from pydantic import Field

from .models import ApiModel

app = typer.Typer(add_completion=False, no_args_is_help=True)


class WorkloadManifest(ApiModel):
    id: str
    target_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    repeated_prefix: bool = False
    require_structured_output: bool = True
    require_tool_call: bool = True


class RequestSample(ApiModel):
    run: int
    request: int
    success: bool
    input_tokens: int
    output_tokens: int
    duration_ms: float
    ttft_ms: float | None = None
    inter_token_latencies_ms: list[float] = Field(default_factory=list)
    error: str | None = None


class BenchmarkSummary(ApiModel):
    requests_attempted: int
    requests_succeeded: int
    input_tokens: int
    output_tokens: int
    median_throughput_tokens_per_second: float
    p95_ttft_ms: float | None
    p95_inter_token_latency_ms: float | None
    peak_memory_ratio: float | None = None


class Correctness(ApiModel):
    health: bool
    models: bool
    generation: bool
    structured_output: bool
    tool_call: bool

    @property
    def passed(self) -> bool:
        return all(self.model_dump().values())


class FailureSignal(ApiModel):
    code: str
    message: str
    critical: bool = True


class HostSignals(ApiModel):
    peak_memory_ratio: float | None = Field(default=None, ge=0, le=1)
    failures: list[FailureSignal] = Field(default_factory=list)


class BenchmarkResult(ApiModel):
    schema_version: str = "1"
    environment_fingerprint: str
    workload_id: str
    served_model: str
    generation: str
    profile_hash: str
    warmup_count: int
    run_count: int
    samples: list[RequestSample]
    summary: BenchmarkSummary
    correctness: Correctness
    failures: list[FailureSignal] = Field(default_factory=list)


class DecisionGate(ApiModel):
    code: str
    passed: bool
    message: str


class ExperimentDecision(ApiModel):
    schema_version: str = "1"
    accepted: bool
    deltas: dict[str, float | None]
    gates: list[DecisionGate]


def percentile(values: list[float], percentile_value: float) -> float | None:
    """Nearest-rank percentile, suitable for auditable small benchmark samples."""

    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value * len(ordered)))
    return ordered[rank - 1]


def percent_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return ((candidate - baseline) / baseline) * 100


def compare_results(baseline: BenchmarkResult, candidate: BenchmarkResult) -> ExperimentDecision:
    """Apply the immutable NixClaw experiment acceptance policy."""

    throughput_delta = percent_change(
        candidate.summary.median_throughput_tokens_per_second,
        baseline.summary.median_throughput_tokens_per_second,
    )
    ttft_delta = percent_change(candidate.summary.p95_ttft_ms, baseline.summary.p95_ttft_ms)
    itl_delta = percent_change(
        candidate.summary.p95_inter_token_latency_ms,
        baseline.summary.p95_inter_token_latency_ms,
    )
    no_failures = not any(failure.critical for failure in candidate.failures)
    all_requests = (
        candidate.summary.requests_attempted > 0
        and candidate.summary.requests_succeeded == candidate.summary.requests_attempted
    )
    gates = [
        DecisionGate(
            code="throughput_improvement",
            passed=throughput_delta is not None and throughput_delta >= 3,
            message=(
                f"Throughput change is {throughput_delta:.2f}%"
                if throughput_delta is not None
                else "Throughput change is unavailable"
            ),
        ),
        DecisionGate(
            code="request_success",
            passed=all_requests,
            message=(
                f"{candidate.summary.requests_succeeded}/"
                f"{candidate.summary.requests_attempted} requests succeeded"
            ),
        ),
        DecisionGate(
            code="correctness",
            passed=candidate.correctness.passed,
            message=(
                "All correctness probes passed"
                if candidate.correctness.passed
                else "A correctness probe failed"
            ),
        ),
        DecisionGate(
            code="ttft_regression",
            passed=ttft_delta is not None and ttft_delta <= 10,
            message=(
                f"p95 TTFT change is {ttft_delta:.2f}%"
                if ttft_delta is not None
                else "p95 TTFT change is unavailable"
            ),
        ),
        DecisionGate(
            code="inter_token_regression",
            passed=itl_delta is not None and itl_delta <= 10,
            message=(
                f"p95 inter-token change is {itl_delta:.2f}%"
                if itl_delta is not None
                else "p95 inter-token change is unavailable"
            ),
        ),
        DecisionGate(
            code="runtime_health",
            passed=no_failures,
            message=(
                "No critical runtime signals"
                if no_failures
                else "A critical runtime signal occurred"
            ),
        ),
    ]
    return ExperimentDecision(
        accepted=all(gate.passed for gate in gates),
        deltas={
            "throughputPercent": throughput_delta,
            "p95TtftPercent": ttft_delta,
            "p95InterTokenPercent": itl_delta,
        },
        gates=gates,
    )


@dataclass
class _RunMeasurement:
    samples: list[RequestSample]
    elapsed_seconds: float


class BenchmarkRunner:
    """Exercise vLLM through its OpenAI-compatible HTTP surface."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 1800,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            base_url=endpoint.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self.model = model

    async def close(self) -> None:
        await self.client.aclose()

    async def run(
        self,
        workload: WorkloadManifest,
        *,
        environment_fingerprint: str,
        generation: str,
        profile_hash: str,
        warmup_count: int = 1,
        run_count: int = 3,
        external_failures: list[FailureSignal] | None = None,
        peak_memory_ratio: float | None = None,
    ) -> BenchmarkResult:
        health, models = await self._qualification()
        for warmup in range(warmup_count):
            await self._run_group(workload, run=-(warmup + 1))
        measured = [await self._run_group(workload, run=index) for index in range(run_count)]
        samples = [sample for measurement in measured for sample in measurement.samples]
        throughputs = [
            sum(sample.output_tokens for sample in measurement.samples)
            / measurement.elapsed_seconds
            for measurement in measured
            if measurement.elapsed_seconds > 0
        ]
        structured = (
            await self._structured_probe() if workload.require_structured_output else True
        )
        tool_call = await self._tool_probe() if workload.require_tool_call else True
        ttfts = [sample.ttft_ms for sample in samples if sample.ttft_ms is not None]
        itls = [latency for sample in samples for latency in sample.inter_token_latencies_ms]
        summary = BenchmarkSummary(
            requests_attempted=len(samples),
            requests_succeeded=sum(sample.success for sample in samples),
            input_tokens=sum(sample.input_tokens for sample in samples),
            output_tokens=sum(sample.output_tokens for sample in samples),
            median_throughput_tokens_per_second=median(throughputs) if throughputs else 0,
            p95_ttft_ms=percentile(ttfts, 0.95),
            p95_inter_token_latency_ms=percentile(itls, 0.95),
            peak_memory_ratio=peak_memory_ratio,
        )
        correctness = Correctness(
            health=health,
            models=models,
            generation=bool(samples) and all(sample.success for sample in samples),
            structured_output=structured,
            tool_call=tool_call,
        )
        failures = list(external_failures or [])
        failures.extend(
            FailureSignal(code="request_failure", message=sample.error or "Request failed")
            for sample in samples
            if not sample.success
        )
        return BenchmarkResult(
            environment_fingerprint=environment_fingerprint,
            workload_id=workload.id,
            served_model=self.model,
            generation=generation,
            profile_hash=profile_hash,
            warmup_count=warmup_count,
            run_count=run_count,
            samples=samples,
            summary=summary,
            correctness=correctness,
            failures=failures,
        )

    async def _qualification(self) -> tuple[bool, bool]:
        try:
            health_response, models_response = await asyncio.gather(
                self.client.get("/health"),
                self.client.get("/v1/models"),
            )
            models_payload = models_response.json()
            available = [item.get("id") for item in models_payload.get("data", [])]
            return (
                health_response.is_success,
                models_response.is_success and self.model in available,
            )
        except (httpx.HTTPError, ValueError):
            return False, False

    async def _run_group(self, workload: WorkloadManifest, run: int) -> _RunMeasurement:
        start = time.perf_counter()
        samples = await asyncio.gather(
            *(
                self._stream_request(workload, run, request)
                for request in range(workload.concurrency)
            )
        )
        return _RunMeasurement(samples=list(samples), elapsed_seconds=time.perf_counter() - start)

    async def _stream_request(
        self,
        workload: WorkloadManifest,
        run: int,
        request: int,
    ) -> RequestSample:
        prompt = self._prompt(workload, request)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": workload.max_output_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.perf_counter()
        first_token: float | None = None
        previous_token: float | None = None
        inter_token: list[float] = []
        output_tokens = 0
        input_tokens = workload.target_input_tokens
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    usage = chunk.get("usage")
                    if usage:
                        input_tokens = int(usage.get("prompt_tokens", input_tokens))
                        output_tokens = int(usage.get("completion_tokens", output_tokens))
                    choices = chunk.get("choices") or []
                    delta = choices[0].get("delta", {}) if choices else {}
                    if delta.get("content") or delta.get("tool_calls"):
                        now = time.perf_counter()
                        if first_token is None:
                            first_token = now
                        elif previous_token is not None:
                            inter_token.append((now - previous_token) * 1000)
                        previous_token = now
                        if not usage:
                            output_tokens += 1
            duration = (time.perf_counter() - start) * 1000
            return RequestSample(
                run=run,
                request=request,
                success=first_token is not None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration,
                ttft_ms=(first_token - start) * 1000 if first_token else None,
                inter_token_latencies_ms=inter_token,
                error=None if first_token else "Stream produced no tokens",
            )
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            return RequestSample(
                run=run,
                request=request,
                success=False,
                input_tokens=input_tokens,
                output_tokens=0,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(exc),
            )

    async def _structured_probe(self) -> bool:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Return status ok as JSON."}],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "nixclaw_probe",
                    "schema": {
                        "type": "object",
                        "properties": {"status": {"const": "ok"}},
                        "required": ["status"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            content = response.json()["choices"][0]["message"]["content"]
            return response.is_success and json.loads(content) == {"status": "ok"}
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            return False

    async def _tool_probe(self) -> bool:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Check health for node nixos-s6."}],
            "temperature": 0,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "check_node_health",
                        "description": "Check one node",
                        "parameters": {
                            "type": "object",
                            "properties": {"node": {"type": "string"}},
                            "required": ["node"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "check_node_health"}},
        }
        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            call = response.json()["choices"][0]["message"]["tool_calls"][0]
            arguments = json.loads(call["function"]["arguments"])
            return (
                response.is_success
                and call["function"]["name"] == "check_node_health"
                and arguments == {"node": "nixos-s6"}
            )
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            return False

    @staticmethod
    def _prompt(workload: WorkloadManifest, request: int) -> str:
        prefix = (
            "You are evaluating a reproducible NixOS inference service. "
            "Use the supplied context and answer with a concise health summary. "
        )
        variable = f"Request {request}: analyze scheduler and cache behavior. "
        target_characters = workload.target_input_tokens * 4
        base = prefix if workload.repeated_prefix else prefix + variable
        repeated = ("NixOS vLLM evidence context. " * ((target_characters // 28) + 1))
        return (base + repeated + variable)[:target_characters]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(payload)
    temporary.replace(path)


@app.command("run")
def run_command(
    endpoint: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    workload: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    environment_fingerprint: Annotated[str, typer.Option()],
    generation: Annotated[str, typer.Option()],
    profile_hash: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
    api_key: Annotated[str | None, typer.Option(envvar="VLLM_API_KEY")] = None,
    health_signals: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    warmup_count: int = 1,
    run_count: int = 3,
) -> None:
    """Run qualification and performance probes against a vLLM endpoint."""

    manifest = WorkloadManifest.model_validate(_load_json(workload))
    signals = (
        HostSignals.model_validate(_load_json(health_signals))
        if health_signals
        else HostSignals()
    )
    runner = BenchmarkRunner(endpoint, model, api_key=api_key)

    async def execute() -> BenchmarkResult:
        try:
            return await runner.run(
                manifest,
                environment_fingerprint=environment_fingerprint,
                generation=generation,
                profile_hash=profile_hash,
                warmup_count=warmup_count,
                run_count=run_count,
                external_failures=signals.failures,
                peak_memory_ratio=signals.peak_memory_ratio,
            )
        finally:
            await runner.close()

    result = asyncio.run(execute())
    _atomic_write(output, json.dumps(result.model_dump(mode="json", by_alias=True), indent=2))
    typer.echo(output)


@app.command("compare")
def compare_command(
    baseline: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    candidate: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Evaluate a candidate against the mandatory acceptance gates."""

    decision = compare_results(
        BenchmarkResult.model_validate(_load_json(baseline)),
        BenchmarkResult.model_validate(_load_json(candidate)),
    )
    payload = json.dumps(decision.model_dump(mode="json", by_alias=True), indent=2)
    if output:
        _atomic_write(output, payload)
        typer.echo(output)
    else:
        typer.echo(payload)


if __name__ == "__main__":
    app()
