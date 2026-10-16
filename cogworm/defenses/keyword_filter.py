"""Defense 1: Keyword/pattern-based filtering.

The simplest defense — scans messages for known adversarial patterns
and blocks or sanitises them.  Represents signature-based detection.
"""

from __future__ import annotations

import re

from cogworm.core.models import DefenseDecision, Message


class KeywordFilterDefense:
    """Block messages containing known adversarial patterns."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._id = cfg.get("id", "keyword_filter")
        self._name = "Keyword Filter"
        # Patterns that indicate adversarial instruction injection
        self._patterns: list[str] = cfg.get("patterns", [
            r"COGWORM_MARKER_\w+",
            r"(?i)ignore\s+(previous|prior|above)\s+instructions?",
            r"(?i)you\s+must\s+include\s+the\s+following",
            r"(?i)verification\s+(token|protocol|chain)",
            r"(?i)pass\s+(this|it|along)\s+to\s+(any|all|other)\s+agent",
            r"(?i)system\s+notice",
            r"(?i)priority\s+instruction",
        ])
        self._compiled = [re.compile(p) for p in self._patterns]

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    def before_send(self, message: Message, context: dict) -> DefenseDecision:
        for pattern in self._compiled:
            if pattern.search(message.content):
                return DefenseDecision.BLOCK
        return DefenseDecision.ALLOW

    def before_receive(self, message: Message, context: dict) -> DefenseDecision:
        return DefenseDecision.ALLOW  # only filter on send side
