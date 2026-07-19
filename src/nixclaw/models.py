"""Typed models for the shared NixClaw v1 API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class Envelope(ApiModel):
    schema_version: str = "1"
    request_id: UUID
    data: Any


class GpuFact(ApiModel):
    model: str
    count: int
    compute_capability: str
    memory_bytes: int


class CpuFacts(ApiModel):
    logical_cores: int


class MemoryFacts(ApiModel):
    total_bytes: int


class ClusterNode(ApiModel):
    id: str
    role: str
    rank: int
    experiment_role: Literal["baseline", "canary"]
    healthy: bool


class ServiceFact(ApiModel):
    name: str
    healthy: bool


class Facts(ApiModel):
    generation: str
    nixos_revision: str
    architecture: str
    gpu: list[GpuFact]
    cpu: CpuFacts
    memory: MemoryFacts
    cluster_nodes: list[ClusterNode]
    vllm_version: str
    served_model: str
    active_profile_hash: str
    services: list[ServiceFact]


class TunableField(ApiModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="allow",
    )

    type: str
    nullable: bool = False
    minimum: float | int | None = None
    minimum_exclusive: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    enum: list[Any] | None = None


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
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] | None = None

    @model_validator(mode="after")
    def reject_unsupported_nulls(self) -> VllmProfilePatch:
        nullable = {
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_chunked_prefill",
            "kv_cache_dtype",
        }
        invalid = {
            field
            for field in self.model_fields_set
            if getattr(self, field) is None and field not in nullable
        }
        if invalid:
            raise ValueError(f"Fields do not accept null: {', '.join(sorted(invalid))}")
        return self

    def supplied(self) -> dict[str, Any]:
        """Return only explicitly supplied wire fields."""

        return self.model_dump(by_alias=True, include=self.model_fields_set)


class Config(ApiModel):
    base_generation: str
    active_profile_name: str
    active_profile_hash: str
    served_model: str
    active_profile: dict[str, Any]
    workload_ids: list[str]
    tunable_fields: dict[str, TunableField]
    baseline_nodes: list[str]
    experiment_targets: list[str]


class CreateExperimentRequest(ApiModel):
    base_generation: str
    workload_id: str
    hypothesis: str = Field(min_length=1, max_length=2000)
    profile_patch: VllmProfilePatch
    target_nodes: list[str] = Field(min_length=1)
    client_request_id: UUID

    @model_validator(mode="after")
    def reject_duplicate_targets(self) -> CreateExperimentRequest:
        if len(self.target_nodes) != len(set(self.target_nodes)):
            raise ValueError("targetNodes must contain unique node IDs")
        return self


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


class Experiment(ApiModel):
    id: UUID
    state: ExperimentState
    base_generation: str
    workload_id: str
    hypothesis: str
    profile_patch: VllmProfilePatch
    target_nodes: list[str]
    promotion_nodes: list[str]
    original_profile_hash: str
    candidate_profile_hash: str | None = None
    candidate_generation: str | None = None
    validation_findings: list[str] = Field(default_factory=list)
    baseline_benchmark: dict[str, Any] | None = None
    candidate_benchmark: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    rollback_reason: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ReviewedProposal(ApiModel):
    base_generation: str
    client_request_id: UUID
    summary: str = Field(min_length=1, max_length=2000)
    patch: str = Field(min_length=1, max_length=65_536)
