"""What the orchestrator and the orders sub-agent agree on, beyond the A2A protocol itself.

A standard A2A skill has no field for an argument schema or for "this changes data", and
the orchestrator needs both to build its tools. They are published in the agent card
through A2A's own extension mechanism, under this URI.

A skill call is a normal A2A message with one data part: {"skill": ..., "params": ...}.
"""

from typing import Any

SKILL_EXTENSION_URI = "urn:shopmind:a2a:skill-schemas:v1"


def restore_integers(value: Any) -> Any:
    """A2A carries structured data as protobuf values, where every number is a float."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: restore_integers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_integers(item) for item in value]
    return value
