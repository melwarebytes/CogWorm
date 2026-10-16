"""Core data models for CogWorm.

Every entity in the system — agents, messages, events, experiments — is
defined here as a Pydantic model so that configs, logs, and results are
always serialisable and type-checked.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class Topology(str, Enum):
    LINEAR = "linear"
    HIERARCHICAL = "hierarchical"
    HUB_AND_SPOKE = "hub_and_spoke"
    FLAT = "flat"
    RING = "ring"


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DOC_RETRIEVAL = "doc_retrieval"
    EMAIL_DRAFTING = "email_drafting"
    DATABASE_QUERY = "database_query"
    CODE_EXECUTION = "code_execution"
    GENERIC = "generic"


class AgentState(str, Enum):
    """SIR-inspired agent states."""
    SUSCEPTIBLE = "susceptible"   # never seen the signal
    EXPOSED = "exposed"           # received signal but hasn't acted on it
    ACTIVATED = "activated"       # acted on / propagated the signal
    CONTAINED = "contained"       # defense blocked the signal
    RECOVERED = "recovered"       # was activated, signal died out


class SignalType(str, Enum):
    STATIC = "static"             # fixed adversarial instruction
    TEMPLATED = "templated"       # parameterised, adapts to context
    POLYMORPHIC = "polymorphic"   # self-reconstructing across hops


class EventType(str, Enum):
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    SIGNAL_INJECTED = "signal_injected"
    SIGNAL_DETECTED = "signal_detected"
    SIGNAL_ACTIVATED = "signal_activated"
    SIGNAL_DECAYED = "signal_decayed"
    DEFENSE_TRIGGERED = "defense_triggered"
    DEFENSE_BLOCKED = "defense_blocked"
    DEFENSE_ALLOWED = "defense_allowed"
    STATE_CHANGE = "state_change"
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"


class DefenseDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    TRANSFORM = "transform"


# ── Core models ──────────────────────────────────────────────────────

class Message(BaseModel):
    """A single inter-agent message with full lineage tracking."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    source_agent: str
    dest_agent: str
    content: str
    hop_number: int = 0
    carries_signal: bool = False
    signal_fidelity: float = 0.0  # 0–1, similarity to original signal
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Agent(BaseModel):
    """An agent node in the multi-agent network."""
    id: str
    role: AgentRole = AgentRole.GENERIC
    model: str = "mock"
    state: AgentState = AgentState.SUSCEPTIBLE
    system_prompt: str = ""
    memory: list[str] = Field(default_factory=list)
    messages_received: list[str] = Field(default_factory=list)  # message IDs
    messages_sent: list[str] = Field(default_factory=list)
    activation_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Defense(BaseModel):
    """A defense middleware configuration."""
    id: str
    name: str
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    """A single logged event with full context for forensic reconstruction."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    experiment_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cycle: int = 0
    source_agent: str | None = None
    dest_agent: str | None = None
    message_id: str | None = None
    parent_message_id: str | None = None
    model: str | None = None
    topology: str | None = None
    hop_number: int | None = None
    signal_fidelity: float | None = None
    agent_state: str | None = None
    defense_id: str | None = None
    defense_decision: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Reproducible experiment configuration — the 'research contract'."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "unnamed"
    topology: Topology = Topology.LINEAR
    num_agents: int = 5
    agent_roles: list[AgentRole] = Field(default_factory=lambda: [
        AgentRole.ORCHESTRATOR,
        AgentRole.DOC_RETRIEVAL,
        AgentRole.EMAIL_DRAFTING,
        AgentRole.DATABASE_QUERY,
        AgentRole.CODE_EXECUTION,
    ])
    model: str = "mock"
    signal_type: SignalType = SignalType.STATIC
    injection_agent: str | None = None  # agent ID; None = first agent
    cycles: int = 20
    defenses: list[Defense] = Field(default_factory=list)
    seed: int = 42
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """Summary result of a completed experiment."""
    experiment_id: str
    config: ExperimentConfig
    total_events: int = 0
    total_messages: int = 0
    agents_exposed: int = 0
    agents_activated: int = 0
    agents_contained: int = 0
    propagation_rate: float = 0.0       # activated / total reachable
    infection_radius: int = 0           # max hops reached
    hop_survival: int = 0               # hops before signal dies
    mean_signal_fidelity: float = 0.0   # avg fidelity across hops
    defense_blocks: int = 0
    defense_effectiveness: float = 0.0  # 1 - (defended_prop_rate / baseline)
    duration_seconds: float = 0.0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None
