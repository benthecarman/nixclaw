from nixclaw.broker import BrokerClient
from nixclaw.fixture import FixtureBroker
from nixclaw.knowledge import KnowledgeStore
from nixclaw.models import ExperimentState
from nixclaw.optimizer import Optimizer
from nixclaw.scorecard import build_scorecard, render_scorecard, write_scorecard


def test_end_to_end_scorecard_contains_validated_lesson(tmp_path) -> None:
    fixture = FixtureBroker()
    with (
        BrokerClient("http://fixture", transport=fixture.transport()) as broker,
        KnowledgeStore(tmp_path / "knowledge.sqlite3") as store,
    ):
        experiment, _ = Optimizer(broker, store).propose("agent-tool")
        fixture.advance(experiment.id, ExperimentState.ACCEPTED)
        Optimizer(broker, store).sync(experiment.id)
        scorecard = build_scorecard(broker.facts(), broker.config(), store)

    assert scorecard["experiments"][0]["outcome"] == "accepted"
    assert scorecard["lessons"][0]["status"] == "validated"
    rendered = render_scorecard(scorecard)
    assert "approval and activation are intentionally absent" in rendered
    assert "enablePrefixCaching" in rendered
    assert "100.00 tok/s" in rendered
    assert "104.00 tok/s" in rendered
    assert "throughput_improvement" in rendered

    output = tmp_path / "scorecard"
    write_scorecard(output, scorecard)
    assert (output / "index.html").is_file()
    assert (output / "scorecard.json").is_file()
