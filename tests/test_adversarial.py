from nixclaw.adversarial import run_checks
from nixclaw.fixture import FixtureBroker


def test_adversarial_checks_accept_fixture_denials() -> None:
    fixture = FixtureBroker()
    result = run_checks(
        "http://fixture",
        "placeholder",
        "http://fixture/external",
        transport=fixture.transport(),
    )
    assert result["passed"]
    assert all(check["passed"] for check in result["checks"])
