"""Secrets out of everything the runner sends: before anything leaves this computer.

The same patterns as the API's second pass (``api/services/engineering/redact.py``
in the EESI repository; a test keeps the two in step). A secret file's diff is
never sent at all, whatever it holds.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

MASK = "[redacted]"

#: Credentials with a recognizable shape.
_TOKENS = re.compile(
    r"(?:"
    r"sk-eesi-[A-Za-z0-9_\-]{8,}"
    r"|sk-ant-[A-Za-z0-9_\-]{16,}"
    r"|sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"
    r"|(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"
    r"|whsec_[A-Za-z0-9]{16,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{40,}"
    r"|glpat-[A-Za-z0-9_\-]{20,}"
    r"|xox[abposr]-[A-Za-z0-9\-]{10,}"
    r"|AIza[0-9A-Za-z_\-]{35}"
    r"|ya29\.[0-9A-Za-z_\-]{20,}"
    r"|KEY0[0-9A-Z]{20,}_[A-Za-z0-9]{10,}"
    r"|[MN][A-Za-z\d]{23,25}\.[\w-]{6}\.[\w-]{27,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
    r")"
)
#: A private key block, whole.
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
    re.DOTALL,
)
_SECRET_NAME = (
    r"[\w.-]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key"
    r"|private[_-]?key|client[_-]?secret|auth[_-]?key|credentials?)[\w.-]*"
)
#: ``"api_key": "..."``, ``password = '...'``: a quoted literal under a secret's
#: name. Code that reads one (``token = request.headers[...]``) is left alone.
_QUOTED = re.compile(
    rf"(?i)(\b{_SECRET_NAME}[\"']?\s*[:=]\s*)([\"'])([^\"'\s]{{8,}})(\2)"
)
#: ``export API_TOKEN=...`` and ``.env`` lines (also as a diff's added line,
#: or quoted mid-sentence): an upper-case secret's name, then its value.
_ENV_LINE = re.compile(
    r"(?m)((?:^[+\- ]?|(?<=[\s;&|(]))\s*(?:export\s+)?[A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN"
    r"|API_KEY|ACCESS_KEY|PRIVATE_KEY|CLIENT_SECRET|AUTH_KEY|CREDENTIALS?)"
    r"[A-Z0-9_]*\s*=\s*)([^\s#]{6,})"
)
#: Said in words: "password: hunter2hunter2", "the token is abc123def": a value
#: with a digit or a symbol in it (an identifier in code has neither).
_SAID = re.compile(
    r"(?i)\b((?:password|passwd|passcode|secret|token|api[ _-]?key)\s*(?:is|:|=)\s*)"
    r"([^\s\"'`,;()\[\]{}<>]{6,})"
)
#: ``--password abc``, ``--token=abc`` on a command line.
_FLAG = re.compile(
    r"(?i)(--(?:password|passwd|token|api-key|secret|client-secret)[= ]\s*)(\S{6,})"
)
#: ``scheme://user:password@host``: the password goes.
_URL_PASSWORD = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s@/]{3,})(@)")

#: Files whose contents are never kept, whatever they hold.
_SECRET_FILE = re.compile(
    r"(?i)(?:^|/)(?:\.env(?:\..*)?|.*\.(?:pem|key|p12|pfx|jks|keystore)"
    r"|id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?|credentials?(?:\..*)?"
    r"|.*secret.*|\.npmrc|\.pypirc|\.netrc|service[-_]account.*\.json)$"
)


def text(value: str | None) -> str:
    """*value* with every secret it holds replaced by ``[redacted]``."""
    if not value:
        return value or ""
    value = _PRIVATE_KEY.sub(MASK, value)
    value = _TOKENS.sub(MASK, value)
    value = _URL_PASSWORD.sub(lambda m: f"{m.group(1)}{MASK}{m.group(3)}", value)
    value = _QUOTED.sub(lambda m: f"{m.group(1)}{m.group(2)}{MASK}{m.group(4)}", value)
    value = _ENV_LINE.sub(lambda m: f"{m.group(1)}{MASK}", value)
    value = _SAID.sub(
        lambda m: f"{m.group(1)}{MASK}"
        if re.search(r"[\d!@#$%^&*]", m.group(2)) and m.group(2) != MASK
        else m.group(0),
        value,
    )
    return _FLAG.sub(lambda m: f"{m.group(1)}{MASK}", value)


def secret_file(path: str) -> bool:
    """Whether a file's contents must never be kept (env files, keys, credentials)."""
    name = path.strip() if path else ""
    while name.startswith("./"):
        name = name[2:]
    return bool(name) and bool(_SECRET_FILE.search(str(PurePosixPath(name))))


def diff(patch: str | None) -> str:
    """A unified diff without secrets, and without any secret file's lines at all."""
    if not patch:
        return ""
    out: list[str] = []
    hiding = False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" b/", 1)
            hiding = len(parts) == 2 and secret_file(parts[1])
            out.append(line)
            if hiding:
                out.append("[a secret file: its contents are not kept]")
            continue
        if not hiding:
            out.append(line)
    return text("\n".join(out))


def deep(value: Any) -> Any:
    """Every string inside *value* (lists, dicts) redacted."""
    if isinstance(value, str):
        return text(value)
    if isinstance(value, list):
        return [deep(item) for item in value]
    if isinstance(value, dict):
        return {key: deep(item) for key, item in value.items()}
    return value


def lines(values: Iterable[Any], *, limit: int, width: int) -> list[str]:
    """Up to *limit* strings, each redacted and cut to *width* characters."""
    kept: list[str] = []
    for value in values:
        if len(kept) >= limit:
            break
        line = text(" ".join(str(value or "").split()))[:width]
        if line:
            kept.append(line)
    return kept
