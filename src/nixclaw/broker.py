"""Restricted HTTP client for the unprivileged NixClaw broker."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
from pydantic import TypeAdapter, ValidationError

from .models import (
    Config,
    CreateExperimentRequest,
    Envelope,
    ErrorEnvelope,
    Experiment,
    Facts,
    ReviewedProposal,
)


class BrokerError(RuntimeError):
    """A stable error returned by the broker or its transport."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}


class BrokerClient:
    """Client exposing only the routes available to the sandboxed agent."""

    def __init__(
        self,
        base_url: str,
        credential: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        headers = {"Accept": "application/json"}
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            follow_redirects=False,
            timeout=timeout or httpx.Timeout(30, connect=3),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BrokerClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def facts(self) -> Facts:
        return self._request("GET", "/v1/facts", Facts)

    def config(self) -> Config:
        return self._request("GET", "/v1/config", Config)

    def experiments(self) -> list[Experiment]:
        return self._request("GET", "/v1/experiments", list[Experiment])

    def experiment(self, experiment_id: UUID | str) -> Experiment:
        return self._request("GET", f"/v1/experiments/{experiment_id}", Experiment)

    def create_experiment(self, request: CreateExperimentRequest) -> Experiment:
        return self._request(
            "POST",
            "/v1/experiments",
            Experiment,
            json=request.model_dump(mode="json", by_alias=True),
            headers={"Idempotency-Key": str(request.client_request_id)},
        )

    def create_proposal(self, proposal: ReviewedProposal) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/proposals",
            dict[str, Any],
            json=proposal.model_dump(mode="json", by_alias=True),
            headers={"Idempotency-Key": str(proposal.client_request_id)},
        )

    def _request(
        self,
        method: str,
        path: str,
        result_type: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BrokerError("transport_error", str(exc)) from exc

        if response.is_redirect:
            raise BrokerError("redirect_rejected", "Broker redirects are not permitted")
        try:
            payload = response.json()
        except ValueError as exc:
            raise BrokerError("invalid_response", "Broker returned non-JSON content") from exc

        if response.is_error:
            try:
                error = ErrorEnvelope.model_validate(payload).error
            except ValidationError as exc:
                raise BrokerError(
                    "invalid_response",
                    "Broker returned an invalid error envelope",
                ) from exc
            raise BrokerError(error.code, error.message, error.details)

        try:
            envelope = Envelope.model_validate(payload)
            return TypeAdapter(result_type).validate_python(envelope.data)
        except ValidationError as exc:
            raise BrokerError(
                "invalid_response",
                "Broker response did not match the v1 contract",
                {"errors": exc.errors(include_url=False)},
            ) from exc
