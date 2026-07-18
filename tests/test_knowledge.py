from nixclaw.broker import BrokerClient
from nixclaw.fixture import FixtureBroker
from nixclaw.knowledge import (
    KnowledgeStore,
    environment_document,
    environment_fingerprint,
)
from nixclaw.models import ExperimentState
from nixclaw.optimizer import CandidateGenerator, Optimizer


def test_environment_fingerprint_is_stable() -> None:
    left = {"b": [2, 1], "a": {"x": True}}
    right = {"a": {"x": True}, "b": [2, 1]}
    assert environment_fingerprint(left) == environment_fingerprint(right)


def test_optimizer_persists_and_retrieves_lesson(tmp_path) -> None:
    fixture = FixtureBroker()
    with (
        BrokerClient("http://fixture", transport=fixture.transport()) as broker,
        KnowledgeStore(tmp_path / "knowledge.sqlite3") as store,
    ):
        experiment, candidate = Optimizer(broker, store).propose("agent-tool")
        assert candidate.patch.enable_prefix_caching is True
        fixture.advance(experiment.id, ExperimentState.ACCEPTED)
        Optimizer(broker, store).sync(experiment.id)
        environment = environment_document(broker.facts(), broker.config(), "agent-tool")
        lessons = store.search_lessons(environment, "agent-tool")
    assert lessons[0].repair == {"enablePrefixCaching": True}
    assert lessons[0].compatibility == "exact"


def test_attempted_candidate_is_not_retried(tmp_path) -> None:
    fixture = FixtureBroker()
    with (
        BrokerClient("http://fixture", transport=fixture.transport()) as broker,
        KnowledgeStore(tmp_path / "knowledge.sqlite3") as store,
    ):
        experiment, _ = Optimizer(broker, store).propose("agent-tool")
        fixture.advance(experiment.id, ExperimentState.REJECTED)
        Optimizer(broker, store).sync(experiment.id)
        facts = broker.facts()
        config = broker.config()
        candidates = CandidateGenerator().generate(facts, config, "agent-tool", store)
    assert candidates
    assert candidates[0].patch.enable_prefix_caching is not True
