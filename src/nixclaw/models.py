"""Typed models for the shared NixClaw v1 API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ApiModel(BaseModel):
    """Strict API model using the repository's camelCase wire convention."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class ErrorDetail(ApiModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(ApiModel):
    schema_version: str = "1"
    request_id: UUID
    error: ErrorDetail


DataT = TypeVar("DataT")


class Envelope(ApiModel):
    schema_version: str = "1"
    request_id: UUID
    data: Any


class GpuFact(ApiModel):
    index: int
    model: str
    compute_capability: str
    memory_bytes: int | None = None


class ResourceFacts(ApiModel):
    cpu_count: int
    memory_bytes: int
    swap_bytes: int


class ClusterNode(ApiModel):
    node_id: str
    role: str
    rank: int
    healthy: bool


class VllmFacts(ApiModel):
    version: str
    model: str
    profile_hash: str
    healthy: bool


class Facts(ApiModel):
    generation: str
    nix_revision: str
    architecture: str
    gpus: list[GpuFact]
    resources: ResourceFacts
    cluster: list[ClusterNode]
    vllm: VllmFacts


class TunableField(ApiModel):
    type: str
    nullable: bool = False
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    values: list[Any] | None = None


class VllmProfilePatch(ApiModel):
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    max_model_len: int | None = Field(default=None, gt=0)
    max_num_seqs: int | None = Field(default=None, gt=0)
    max_num_batched_tokens: int | None = Field(default=None, gt=0)
    tensor_parallel_size: int | None = Field(default=None, gt=0)
    pipeline_parallel_size: int | None = Field(default=None, gt=0)
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    enforce_eager: bool | None = None
    kv_cache_dtype: str | None = None

    def supplied(self) -> dict[str, Any]:
        """Return only explicitly supplied wire fields."""

        return self.model_dump(by_alias=True, exclude_none=True)


class Config(ApiModel):
    base_generation: str
    active_profile_name: str
    profile_hash: str
    served_model: str
    active_profile: dict[str, Any]
    workload_ids: list[str]
    tunables: dict[str, TunableField]


class CreateExperimentRequest(ApiModel):
    base_generation: str
    workload_id: str
    hypothesis: str = Field(min_length=1, max_length=1000)
    profile_patch: VllmProfilePatch
    client_request_id: UUID


class ExperimentState(StrEnum):
    SUBMITTED = "submitted"
    VALIDATING = "validating"
    BUILT = "built"
    AWAITING_APPROVAL = "awaitingApproval"
    ACTIVE = "active"
    MEASURING = "measuring"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolledBack"
    FAILED = "failed"


TERMINAL_EXPERIMENT_STATES = {
    ExperimentState.ACCEPTED,
    ExperimentState.REJECTED,
    ExperimentState.ROLLED_BACK,
    ExperimentState.FAILED,
}


class ValidationFinding(ApiModel):
    code: str
    message: str
    field: str | None = None


class Experiment(ApiModel):
    id: UUID
    state: ExperimentState
    base_generation: str
    workload_id: str
    hypothesis: str
    profile_patch: VllmProfilePatch
    client_request_id: UUID
    original_profile_hash: str
    candidate_profile_hash: str | None = None
    candidate_generation: str | None = None
    findings: list[ValidationFinding] = Field(default_factory=list)
    baseline_result: dict[str, Any] | None = None
    candidate_result: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    rollback_reason: str | None = None
    created_at: datetime
    updated_at: datetime


class ReviewedProposal(ApiModel):
    base_generation: str
    summary: str = Field(min_length=1, max_length=1000)
    diff: str = Field(min_length=1, max_length=131_072)
    client_request_id: UUID
