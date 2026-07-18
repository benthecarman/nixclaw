from pathlib import Path

from jsonschema.validators import validator_for

SCHEMA_ROOT = Path(__file__).parents[1] / "schemas" / "nixclaw" / "v1"


def test_all_json_schemas_are_valid() -> None:
    schemas = list(SCHEMA_ROOT.glob("*.schema.json"))
    assert schemas
    for schema_path in schemas:
        schema = __import__("json").loads(schema_path.read_text())
        validator_for(schema).check_schema(schema)
