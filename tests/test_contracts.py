import json
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from jsonschema.validators import validator_for
from referencing import Registry, Resource

from nixclaw.fixture import FixtureBroker
from nixclaw.models import ExperimentState

SCHEMA_ROOT = Path(__file__).parents[1] / "schemas" / "nixclaw" / "v1"


def test_all_json_schemas_are_valid() -> None:
    schemas = list(SCHEMA_ROOT.glob("*.schema.json"))
    assert schemas
    for schema_path in schemas:
        schema = __import__("json").loads(schema_path.read_text())
        validator_for(schema).check_schema(schema)


def test_fixture_facts_and_config_match_canonical_schemas() -> None:
    resources = []
    schemas = {}
    for schema_path in SCHEMA_ROOT.glob("*.schema.json"):
        schema = json.loads(schema_path.read_text())
        schemas[schema_path.name] = schema
        resources.append((schema["$id"], Resource.from_contents(schema)))
    registry = Registry().with_resources(resources)
    fixture = FixtureBroker()
    for route, schema_name in (
        ("/v1/facts", "facts-response.schema.json"),
        ("/v1/config", "config-response.schema.json"),
    ):
        response = fixture.handle(httpx.Request("GET", f"http://fixture{route}"))
        validator = validator_for(schemas[schema_name])(
            schemas[schema_name],
            registry=registry,
        )
        validator.validate(response.json())


def test_fixture_experiment_matches_canonical_schema() -> None:
    schemas = {}
    resources = []
    for schema_path in SCHEMA_ROOT.glob("*.schema.json"):
        schema = json.loads(schema_path.read_text())
        schemas[schema_path.name] = schema
        resources.append((schema["$id"], Resource.from_contents(schema)))
    registry = Registry().with_resources(resources)
    fixture = FixtureBroker()
    created = fixture.handle(
        httpx.Request(
            "POST",
            "http://fixture/v1/experiments",
            json={
                "baseGeneration": fixture.generation,
                "workloadId": "agent-tools",
                "hypothesis": "Repeated prefixes should benefit from caching.",
                "profilePatch": {"enablePrefixCaching": True},
                "clientRequestId": str(uuid4()),
            },
        )
    ).json()["data"]
    fixture.advance(UUID(created["id"]), ExperimentState.ACCEPTED)
    accepted = fixture.handle(
        httpx.Request("GET", f"http://fixture/v1/experiments/{created['id']}")
    ).json()["data"]
    validator = validator_for(schemas["experiment.schema.json"])(
        schemas["experiment.schema.json"],
        registry=registry,
    )
    validator.validate(accepted)
