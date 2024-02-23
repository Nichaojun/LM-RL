import json
import pytest
import jsonschema
import glob


@pytest.mark.parametrize("api_file", glob.glob("turtlesim_msgs/srv/*.json"))
def test_validate_schema(api_file):
    print(f"Validating {api_file}...")
    api = None
    with open(api_file, "r") as f:
        api = json.load(f)

    jsonschema.Draft7Validator.check_schema(api)
