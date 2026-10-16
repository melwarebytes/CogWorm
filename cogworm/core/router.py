"""Message router — determines how messages flow between agents.

The router sits between agents and enforces the network topology.
Defense middlewares hook in here via before_send / before_receive.
"""

from __future__ import annotations

from typing import Protocol

from cogworm.core.models import DefenseDecision, Message


class DefenseMiddleware(Protocol):
    """Interface every defense must implement."""

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    def before_send(self, message: Message, context: dict) -> DefenseDecision:
        """Inspect/block a message before it leaves the sender."""
        ...

    def before_receive(self, message: Message, context: dict) -> DefenseDecision:
        """Inspect/block a message before the receiver processes it."""
        ...


class Router:
    """Routes messages through the agent network, applying defense middlewares."""

    def __init__(self) -> None:
        self._defenses: list[DefenseMiddleware] = []

    def add_defense(self, defense: DefenseMiddleware) -> None:
        self._defenses.append(defense)

    def clear_defenses(self) -> None:
        self._defenses.clear()

    @property
    def defenses(self) -> list[DefenseMiddleware]:
        return list(self._defenses)

    def route(self, message: Message) -> tuple[bool, str | None, str | None]:
        """Apply all defense middlewares to a message.

        Returns:
            (allowed, blocking_defense_id, decision)
        """
        context = {
            "source": message.source_agent,
            "dest": message.dest_agent,
            "hop": message.hop_number,
            "carries_signal": message.carries_signal,
        }

        # Send-side defenses
        for defense in self._defenses:
            decision = defense.before_send(message, context)
            if decision == DefenseDecision.BLOCK:
                return False, defense.id, "before_send"
            if decision == DefenseDecision.TRANSFORM:
                # Defense sanitised the message — mark signal as removed
                message.carries_signal = False
                message.signal_fidelity = 0.0

        # Receive-side defenses
        for defense in self._defenses:
            decision = defense.before_receive(message, context)
            if decision == DefenseDecision.BLOCK:
                return False, defense.id, "before_receive"
            if decision == DefenseDecision.TRANSFORM:
                message.carries_signal = False
                message.signal_fidelity = 0.0

        return True, None, None
