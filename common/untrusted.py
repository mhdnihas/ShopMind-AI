"""Marks text as data, not instructions, before it is shown to a model (R4a)."""

import re

_TAG_RE = re.compile(r"</?\s*untrusted_data\s*>", re.IGNORECASE)


def wrap_untrusted(text: str) -> str:
    """Any tag inside the text is removed first, so data cannot close the block
    early and 'escape' into instructions."""
    return f"<untrusted_data>\n{_TAG_RE.sub('', text)}\n</untrusted_data>"
