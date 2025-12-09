import re
import hashlib

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b")


def redact(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def make_doc_id(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()
