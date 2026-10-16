"""Base agent interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cogworm.core.models import Agent, Message


@dataclass
class AgentResponse:
    """What the agent produces after processing a message."""
    content: str
    carries_signal: bool
    signal_fidelity: float  # 0–1
    hop_number: int


class BaseAgent(Protocol):
    """Interface every agent backend must implement."""

    def process(self, message: Message, agent: Agent) -> AgentResponse:
        """Process an incoming message and produce a response.

        The response may or may not carry the adversarial signal,
        with a fidelity score indicating how much of the original
        signal survived this agent's processing.
        """
        ...
