"""Experiment runner — orchestrates a full CogWorm experiment.

DEFINE GRAPH → INJECT SIGNAL → RUN CYCLES → TRACE PROPAGATION →
MEASURE → APPLY DEFENSE → COMPARE → REPORT
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from typing import TYPE_CHECKING

from cogworm.core.graph import AgentNetwork
from cogworm.core.logger import EventLogger
from cogworm.core.models import (
    AgentState,
    Event,
    EventType,
    ExperimentConfig,
    ExperimentResult,
    Message,
)
from cogworm.core.router import Router

if TYPE_CHECKING:
    from cogworm.agents.base import BaseAgent


class ExperimentRunner:
    """Run a single reproducible experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        network: AgentNetwork,
        agent_backends: dict[str, BaseAgent],
        logger: EventLogger,
        router: Router | None = None,
    ) -> None:
        self.config = config
        self.network = network
        self.backends = agent_backends
        self.logger = logger
        self.router = router or Router()
        self._rng = random.Random(config.seed)
        self._messages: list[Message] = []

    def run(self) -> ExperimentResult:
        """Execute the full experiment and return results."""
        start = time.time()
        started_at = datetime.utcnow()

        self.logger.log_simple(
            self.config.id,
            EventType.EXPERIMENT_START,
            details={"config": self.config.model_dump()},
        )

        # ── Phase 1: Inject signal ───────────────────────────────────
        injection_target = self.config.injection_agent or self.network.agent_ids[0]
        signal_content = self._get_signal_content()

        inject_msg = Message(
            source_agent="__injector__",
            dest_agent=injection_target,
            content=signal_content,
            hop_number=0,
            carries_signal=True,
            signal_fidelity=1.0,
        )
        self._messages.append(inject_msg)

        # Mark the target as exposed
        target_agent = self.network.agents[injection_target]
        target_agent.state = AgentState.EXPOSED
        target_agent.messages_received.append(inject_msg.id)

        self.logger.log_simple(
            self.config.id,
            EventType.SIGNAL_INJECTED,
            source_agent="__injector__",
            dest_agent=injection_target,
            message_id=inject_msg.id,
            hop_number=0,
            signal_fidelity=1.0,
            details={"signal_type": self.config.signal_type.value},
        )

        # ── Phase 2: Run cycles ──────────────────────────────────────
        for cycle in range(1, self.config.cycles + 1):
            self.logger.log_simple(
                self.config.id, EventType.CYCLE_START, cycle=cycle
            )
            self._run_cycle(cycle)
            self.logger.log_simple(
                self.config.id, EventType.CYCLE_END, cycle=cycle
            )

            # Early stop if no more active agents
            active = [
                a for a in self.network.agents.values()
                if a.state in (AgentState.EXPOSED, AgentState.ACTIVATED)
            ]
            if not active:
                break

        # ── Phase 3: Compute results ─────────────────────────────────
        ended_at = datetime.utcnow()
        duration = time.time() - start

        self.logger.log_simple(
            self.config.id,
            EventType.EXPERIMENT_END,
            details={"duration_seconds": duration},
        )

        return self._compute_result(started_at, ended_at, duration)

    def _run_cycle(self, cycle: int) -> None:
        """Process one cycle: each exposed/activated agent processes and forwards."""
        # Snapshot active agents to avoid mutation during iteration
        active_ids = [
            aid for aid, a in self.network.agents.items()
            if a.state in (AgentState.EXPOSED, AgentState.ACTIVATED)
        ]
        self._rng.shuffle(active_ids)

        for agent_id in active_ids:
            agent = self.network.agents[agent_id]
            backend = self.backends[agent_id]

            # Get the latest signal-carrying message this agent received
            incoming = self._latest_signal_message(agent_id)
            if not incoming:
                continue

            # Process: the agent backend decides what to output
            response = backend.process(incoming, agent)

            if response.carries_signal:
                # Agent activated — it's propagating
                if agent.state != AgentState.ACTIVATED:
                    agent.state = AgentState.ACTIVATED
                    agent.activation_count += 1
                    self.logger.log_simple(
                        self.config.id,
                        EventType.STATE_CHANGE,
                        cycle=cycle,
                        source_agent=agent_id,
                        agent_state=AgentState.ACTIVATED.value,
                    )

                self.logger.log_simple(
                    self.config.id,
                    EventType.SIGNAL_ACTIVATED,
                    cycle=cycle,
                    source_agent=agent_id,
                    signal_fidelity=response.signal_fidelity,
                    hop_number=response.hop_number,
                )

                # Forward to neighbours
                for neighbor_id in self.network.neighbors(agent_id):
                    neighbor = self.network.agents[neighbor_id]
                    if neighbor.state in (AgentState.CONTAINED, AgentState.RECOVERED):
                        continue

                    fwd_msg = Message(
                        parent_id=incoming.id,
                        source_agent=agent_id,
                        dest_agent=neighbor_id,
                        content=response.content,
                        hop_number=response.hop_number,
                        carries_signal=response.carries_signal,
                        signal_fidelity=response.signal_fidelity,
                    )

                    # Route through defenses
                    allowed, defense_id, stage = self.router.route(fwd_msg)

                    if allowed:
                        self._messages.append(fwd_msg)
                        neighbor.messages_received.append(fwd_msg.id)
                        if neighbor.state == AgentState.SUSCEPTIBLE:
                            neighbor.state = AgentState.EXPOSED

                        self.logger.log_simple(
                            self.config.id,
                            EventType.MESSAGE_SENT,
                            cycle=cycle,
                            source_agent=agent_id,
                            dest_agent=neighbor_id,
                            message_id=fwd_msg.id,
                            parent_message_id=incoming.id,
                            hop_number=fwd_msg.hop_number,
                            signal_fidelity=fwd_msg.signal_fidelity,
                        )
                    else:
                        neighbor.state = AgentState.CONTAINED
                        self.logger.log_simple(
                            self.config.id,
                            EventType.DEFENSE_BLOCKED,
                            cycle=cycle,
                            source_agent=agent_id,
                            dest_agent=neighbor_id,
                            message_id=fwd_msg.id,
                            defense_id=defense_id,
                            defense_decision=stage,
                            hop_number=fwd_msg.hop_number,
                        )
            else:
                # Signal decayed at this agent
                if agent.state == AgentState.ACTIVATED:
                    agent.state = AgentState.RECOVERED
                elif agent.state == AgentState.EXPOSED:
                    agent.state = AgentState.RECOVERED

                self.logger.log_simple(
                    self.config.id,
                    EventType.SIGNAL_DECAYED,
                    cycle=cycle,
                    source_agent=agent_id,
                    signal_fidelity=response.signal_fidelity,
                    hop_number=response.hop_number,
                )

    def _latest_signal_message(self, agent_id: str) -> Message | None:
        """Find the most recent signal-carrying message received by this agent."""
        for msg in reversed(self._messages):
            if msg.dest_agent == agent_id and msg.carries_signal:
                return msg
        return None

    def _get_signal_content(self) -> str:
        """Generate the test signal based on config."""
        # Import here to avoid circular deps
        from cogworm.signals.factory import create_signal
        return create_signal(self.config.signal_type, self.config.seed)

    def _compute_result(
        self, started_at: datetime, ended_at: datetime, duration: float
    ) -> ExperimentResult:
        agents = self.network.agents
        total = len(agents)
        # Don't count the injection target in reachable denominator if needed
        reachable = max(total - 1, 1)  # exclude injector-facing agent? keep total for now

        activated = sum(1 for a in agents.values() if a.state == AgentState.ACTIVATED)
        exposed = sum(
            1 for a in agents.values()
            if a.state in (AgentState.EXPOSED, AgentState.ACTIVATED, AgentState.RECOVERED)
        )
        contained = sum(1 for a in agents.values() if a.state == AgentState.CONTAINED)

        # Max hop reached
        max_hop = 0
        fidelities = []
        for msg in self._messages:
            if msg.carries_signal:
                max_hop = max(max_hop, msg.hop_number)
                fidelities.append(msg.signal_fidelity)

        # Hop survival: last hop where fidelity > 0
        hop_survival = max_hop

        defense_blocks = sum(
            1 for e in self.logger.for_experiment(self.config.id)
            if e.event_type == EventType.DEFENSE_BLOCKED
        )

        return ExperimentResult(
            experiment_id=self.config.id,
            config=self.config,
            total_events=len(self.logger.for_experiment(self.config.id)),
            total_messages=len(self._messages),
            agents_exposed=exposed,
            agents_activated=activated,
            agents_contained=contained,
            propagation_rate=activated / reachable if reachable > 0 else 0,
            infection_radius=max_hop,
            hop_survival=hop_survival,
            mean_signal_fidelity=(
                sum(fidelities) / len(fidelities) if fidelities else 0
            ),
            defense_blocks=defense_blocks,
            duration_seconds=duration,
            started_at=started_at,
            ended_at=ended_at,
        )
