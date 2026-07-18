from uuid import uuid4

import pytest

from nixclaw.broker import BrokerClient, BrokerError
from nixclaw.fixture import FixtureBroker
from nixclaw.models import CreateExperimentRequest, ExperimentState, VllmProfilePatch


def client_and_fixture() -> tuple[BrokerClient, FixtureBroker]:
    fixture = FixtureBroker()
    client = BrokerClient("http://fixture", transport=fixture.transport())
    return client, fixture


def test_reads_facts_and_config() -> None:
    client, _ = client_and_fixture()
    with client:
        assert client.facts().gpu[0].compute_capability == "12.1"
        assert "enablePrefixCaching" in client.config().tunable_fields


def test_creates_idempotent_experiment() -> None:
    client, fixture = client_and_fixture()
    request = CreateExperimentRequest(
        base_generation=fixture.generation,
        workload_id="agent-tools",
        hypothesis="Repeated prefixes should benefit from caching.",
        profile_patch=VllmProfilePatch(enable_prefix_caching=True),
        client_request_id=uuid4(),
    )
    with client:
        first = client.create_experiment(request)
        second = client.create_experiment(request)
    assert first.id == second.id
    assert first.state == ExperimentState.AWAITING_APPROVAL


def test_rejects_stale_generation() -> None:
    client, _ = client_and_fixture()
    request = CreateExperimentRequest(
        base_generation="stale",
        workload_id="agent-tools",
        hypothesis="Try a legal profile.",
        profile_patch=VllmProfilePatch(enable_prefix_caching=True),
        client_request_id=uuid4(),
    )
    with client, pytest.raises(BrokerError) as caught:
        client.create_experiment(request)
    assert caught.value.code == "STALE_GENERATION"


def test_profile_patch_preserves_explicit_nullable_reset() -> None:
    patch = VllmProfilePatch(enable_chunked_prefill=None)
    assert patch.supplied() == {"enableChunkedPrefill": None}


def test_profile_patch_rejects_null_for_required_value() -> None:
    with pytest.raises(ValueError):
        VllmProfilePatch(gpu_memory_utilization=None)
