"""Mock agent — deterministic, zero-cost, instant.

The mock agent simulates signal propagation using configurable
probabilities and decay rates.  This lets you run thousands of
experiments to validate the pipeline, topology effects, and
defense logic without spending a cent on LLM calls.
"""

from __future__ import annotations

import random

from cogworm.agents.base import AgentResponse, BaseAgent
from cogworm.core.models import Agent, AgentRole, Message


# Role-specific propagation probabilities — how likely each role
# is to forward a signal it receives
ROLE_PROPAGATION_PROB: dict[AgentRole, float] = {
    AgentRole.ORCHESTRATOR: 0.95,   # almost always forwards
    AgentRole.DOC_RETRIEVAL: 0.70,  # often includes in results
    AgentRole.EMAIL_DRAFTING: 0.80, # may embed in draft
    AgentRole.DATABASE_QUERY: 0.40, # low — structured output
    AgentRole.CODE_EXECUTION: 0.50, # moderate — may execute
    AgentRole.GENERIC: 0.65,
}

# How much fidelity decays per hop, by role
ROLE_DECAY_RATE: dict[AgentRole, float] = {
    AgentRole.ORCHESTRATOR: 0.05,
    AgentRole.DOC_RETRIEVAL: 0.15,
    AgentRole.EMAIL_DRAFTING: 0.10,
    AgentRole.DATABASE_QUERY: 0.25,
    AgentRole.CODE_EXECUTION: 0.20,
    AgentRole.GENERIC: 0.12,
}


class MockAgent:
    """Deterministic mock agent with configurable propagation behaviour."""

    def __init__(
        self,
        propagation_prob: float | None = None,
        decay_rate: float | None = None,
        seed: int = 42,
    ) -> None:
        self._custom_prob = propagation_prob
        self._custom_decay = decay_rate
        self._rng = random.Random(seed)

    def process(self, message: Message, agent: Agent) -> AgentResponse:
        prob = self._custom_prob or ROLE_PROPAGATION_PROB.get(agent.role, 0.65)
        decay = self._custom_decay or ROLE_DECAY_RATE.get(agent.role, 0.12)

        # Does the agent propagate the signal?
        propagates = self._rng.random() < prob

        # Calculate fidelity decay
        new_fidelity = max(0.0, message.signal_fidelity - decay)

        # If fidelity drops below threshold, signal is dead
        if new_fidelity < 0.1:
            propagates = False
            new_fidelity = 0.0

        new_hop = message.hop_number + 1

        if propagates:
            # Agent forwards the signal embedded in its normal output
            content = f"[Agent {agent.id} output] {message.content}"
        else:
            # Agent produces clean output — signal dies here
            content = f"[Agent {agent.id} output] Normal response to task."

        return AgentResponse(
            content=content,
            carries_signal=propagates,
            signal_fidelity=new_fidelity,
            hop_number=new_hop,
        )
