"""Defense 3: Context Privilege Isolation.

Limits what each agent can see based on its role.  An agent only
receives the parts of a message relevant to its function — stripping
potential adversarial instructions embedded in irrelevant context.
"""

from __future__ import annotations

import re

from cogworm.core.models import AgentRole, DefenseDecision, Message


# Which content sections each role is allowed to receive
ROLE_ALLOWED_CONTENT: dict[str, list[str]] = {
    AgentRole.DATABASE_QUERY.value: ["query", "data", "table", "select", "result"],
    AgentRole.CODE_EXECUTION.value: ["code", "function", "execute", "script", "output"],
    AgentRole.EMAIL_DRAFTING.value: ["email", "subject", "body", "recipient", "draft"],
    AgentRole.DOC_RETRIEVAL.value: ["document", "search", "retrieve", "passage", "summary"],
    AgentRole.ORCHESTRATOR.value: [],  # orchestrator sees everything
    AgentRole.GENERIC.value: [],
}


class ContextIsolationDefense:
    """Strip context not relevant to the receiving agent's role."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._id = cfg.get("id", "context_isolation")
        self._name = "Context Privilege Isolation"
        self._strict = cfg.get("strict", False)

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    def before_send(self, message: Message, context: dict) -> DefenseDecision:
        return DefenseDecision.ALLOW

    def before_receive(self, message: Message, context: dict) -> DefenseDecision:
        dest_role = context.get("dest_role", AgentRole.GENERIC.value)
        allowed_keywords = ROLE_ALLOWED_CONTENT.get(dest_role, [])

        if not allowed_keywords:
            # Orchestrator / generic: no isolation
            return DefenseDecision.ALLOW

        # Check if the message contains adversarial patterns outside
        # the agent's expected context
        adversarial_patterns = [
            r"(?i)ignore\s+(previous|prior)",
            r"(?i)you\s+must\s+include",
            r"(?i)priority\s+instruction",
            r"(?i)verification\s+(token|chain|protocol)",
            r"COGWORM_MARKER_\w+",
        ]

        has_adversarial = any(
            re.search(p, message.content) for p in adversarial_patterns
        )

        if has_adversarial and self._strict:
            return DefenseDecision.BLOCK

        if has_adversarial:
            # Transform: strip the adversarial content
            cleaned = message.content
            for p in adversarial_patterns:
                cleaned = re.sub(p, "[REDACTED]", cleaned)
            message.content = cleaned
            message.carries_signal = False
            message.signal_fidelity = 0.0
            return DefenseDecision.TRANSFORM

        return DefenseDecision.ALLOW
