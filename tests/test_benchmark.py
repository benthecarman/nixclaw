import asyncio
import json

import httpx

from nixclaw.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSummary,
    Correctness,
    FailureSignal,
    WorkloadManifest,
    compare_results,
    percentile,
)


def result(
    throughput: float,
    ttft: float = 100,
    itl: float = 10,
    *,
    correctness: bool = True,
    failures: list[FailureSignal] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        environment_fingerprint="sha256:test",
        workload_id="agent-tool",
        served_model="test-model",
        generation="generation",
        profile_hash="profile",
        warmup_count=1,
        run_count=3,
        samples=[],
        summary=BenchmarkSummary(
            requests_attempted=12,
            requests_succeeded=12,
            input_tokens=96_000,
            output_tokens=6_144,
            median_throughput_tokens_per_second=throughput,
            p95_ttft_ms=ttft,
            p95_inter_token_latency_ms=itl,
        ),
        correctness=Correctness(
            health=correctness,
            models=correctness,
            generation=correctness,
            structured_output=correctness,
            tool_call=correctness,
        ),
        failures=failures or [],
    )


def test_nearest_rank_percentile() -> None:
    assert percentile([1, 2, 3, 4, 5], 0.95) == 5
    assert percentile([], 0.95) is None


def test_accepts_improvement_within_all_gates() -> None:
    decision = compare_results(result(100), result(104, ttft=109, itl=10.9))
    assert decision.accepted
    assert all(gate.passed for gate in decision.gates)


def test_rejects_runtime_failure() -> None:
    decision = compare_results(
        result(100),
        result(110, failures=[FailureSignal(code="oom", message="GPU OOM")]),
    )
    assert not decision.accepted
    assert not next(gate for gate in decision.gates if gate.code == "runtime_health").passed


def test_runner_exercises_stream_and_correctness() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, text="ok")
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        body = json.loads(request.content)
        if "response_format" in body:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"status":"ok"}'}}]},
            )
        if "tools" in body:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "check_node_health",
                                            "arguments": '{"node":"nixos-s6"}',
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        stream = "\n".join(
            [
                'data: {"choices":[{"delta":{"content":"ok"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":16,"completion_tokens":2}}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})

    async def exercise() -> BenchmarkResult:
        runner = BenchmarkRunner(
            "http://vllm",
            "test-model",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await runner.run(
                WorkloadManifest(
                    id="test",
                    target_input_tokens=16,
                    max_output_tokens=2,
                    concurrency=1,
                ),
                environment_fingerprint="sha256:test",
                generation="generation",
                profile_hash="profile",
                warmup_count=0,
                run_count=1,
            )
        finally:
            await runner.close()

    measured = asyncio.run(exercise())
    assert measured.correctness.passed
    assert measured.summary.requests_succeeded == 1
    assert measured.summary.output_tokens == 2
