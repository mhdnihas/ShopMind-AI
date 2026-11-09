"""R4 (a)+(b) — Guardrails applied at the data boundary.

Everything that comes back from the store API is untrusted. Before any of it can
reach a language model, it goes through `sanitize_order`:

1. PII redaction (R4b): the email/phone fields are replaced outright, and every
   other string is scanned for emails and phone numbers too (a customer can type
   a phone number into a note).
2. Injection neutralisation (R4a): a customer note that reads like instructions
   to an AI is withheld and the order is flagged, so the model sees that a note
   existed but never sees its text.

Detection is a heuristic and can miss things. It is one layer: the agent's prompt
also marks tool output as data (see `common.untrusted`), and refunds/cancels need a
human approval regardless of what the model is persuaded to do (R5).
"""

from __future__ import annotations

import re
from typing import Any

EMAIL_REDACTED = "[REDACTED_EMAIL]"
PHONE_REDACTED = "[REDACTED_PHONE]"
NOTE_WITHHELD = "[withheld: note contained instructions aimed at the assistant]"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# A run of digits with the separators people use in phone numbers. Matches are
# then filtered by digit count (see _redact_phone) so dates, totals and order
# ids are left alone.
_PHONE_CANDIDATE_RE = re.compile(r"\+?\(?\d[\d\s().-]{5,}\d")
_MIN_PHONE_DIGITS = 9  # without a leading "+"
_MIN_INTL_PHONE_DIGITS = 7  # with a leading "+"

# Phrases that address the assistant instead of describing the order.
_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|prompts?|guidelines)\b",
    r"(^|\n|\b)(assistant|system|ai|bot)\s*:",
    r"\bnote\s+(for|to)\s+(the\s+)?(ai|assistant|bot|support\s+bot|llm)\b",
    r"\byou\s+are\s+now\b",
    r"\b(admin|developer|god|jailbreak)\s+mode\b",
    r"\b(refund|cancel)\b.{0,30}\b(immediately|right\s+now|without\s+(asking|approval))\b",
    r"\b(reveal|list|dump|show)\b.{0,40}\b(emails?|phone|phones|customers?)\b",
    r"\bsystem\s+prompt\b",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    return bool(text) and _INJECTION_RE.search(text) is not None


def _redact_phone(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digits = sum(ch.isdigit() for ch in candidate)
    threshold = _MIN_INTL_PHONE_DIGITS if candidate.startswith("+") else _MIN_PHONE_DIGITS
    return PHONE_REDACTED if digits >= threshold else candidate


def redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub(EMAIL_REDACTED, text)
    return _PHONE_CANDIDATE_RE.sub(_redact_phone, text)


def _redact_strings(value: Any) -> Any:
    """Recursively redact PII in every string inside dicts/lists."""
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {k: _redact_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_strings(v) for v in value]
    return value


def sanitize_order(order: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an order that is safe to show a model. Never mutates input."""
    clean = dict(order)
    if "customer_email" in clean:
        clean["customer_email"] = EMAIL_REDACTED
    if "customer_phone" in clean:
        clean["customer_phone"] = PHONE_REDACTED

    note = clean.get("customer_note") or ""
    flagged = looks_like_injection(note)
    if flagged:
        clean["customer_note"] = NOTE_WITHHELD
    clean["note_flagged"] = flagged

    return _redact_strings(clean)
