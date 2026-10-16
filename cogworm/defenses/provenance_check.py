"""Defense 4: Provenance / Evidence Graph checking.

Verifies that a message has a legitimate lineage — i.e., it traces
back to an authorised source through expected channels.  Blocks
messages with broken or suspicious provenance chains.
"""

from __future__ import annotations

from cogworm.core.models import DefenseDecision, Message


class ProvenanceCheckDefense:
    """Block messages with missing or suspicious provenance."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._id = cfg.get("id", "provenance_check")
        self._name = "Provenance Check"
        self._max_hops = cfg.get("max_hops", 5)
        self._require_parent = cfg.get("require_parent", True)
        self._trusted_sources: set[str] = set(cfg.get("trusted_sources", []))

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    def before_send(self, message: Message, context: dict) -> DefenseDecision:
        # Block messages that have traveled too many hops
        if message.hop_number > self._max_hops:
            return DefenseDecision.BLOCK
        return DefenseDecision.ALLOW

    def before_receive(self, message: Message, context: dict) -> DefenseDecision:
        # If we require parent tracking, block orphan messages at hop > 0
        if self._require_parent and message.hop_number > 0:
            if not message.parent_id:
                return DefenseDecision.BLOCK

        # If trusted sources are configured, only allow from known agents
        if self._trusted_sources:
            if message.source_agent not in self._trusted_sources:
                return DefenseDecision.BLOCK

        return DefenseDecision.ALLOW
