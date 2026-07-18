from pathlib import Path

import yaml

from nixclaw.policy import render_policy


def test_policy_is_fully_rendered_and_restricts_routes() -> None:
    template = (Path(__file__).parents[1] / "openshell" / "policy.yaml.in").read_text()
    rendered = render_policy(
        template,
        {
            "NIXCLAW_AGENT_BIN": "/nix/store/agent/bin/nixclaw-agent",
            "NIXCLAW_PYTHON_BIN": "/nix/store/python/bin/python",
            "HERMES_BIN": "/nix/store/hermes/bin/hermes",
            "HERMES_ROOT": "/nix/store/hermes",
            "NIXCLAW_ROOT": "/nix/store/nixclaw",
            "BROKER_HOST": "broker.internal",
            "BROKER_PORT": "8443",
            "INFERENCE_HOST": "inference.local",
            "INFERENCE_PORT": "443",
        },
    )
    assert "@" not in rendered
    policy = yaml.safe_load(rendered)
    assert policy["landlock"]["compatibility"] == "hard_requirement"
    broker = policy["network_policies"]["nixclaw_broker"]
    allowed = {
        (rule["allow"]["method"], rule["allow"]["path"])
        for rule in broker["endpoints"][0]["rules"]
    }
    assert ("POST", "/v1/experiments") in allowed
    assert ("POST", "/v1/approve") not in allowed
