"""Agent network graph — topology-independent representation of the MAS.

The graph layer is fully independent from the LLM layer.  Every topology
is just a different way to wire the same Agent nodes.
"""

from __future__ import annotations

import networkx as nx

from cogworm.core.models import Agent, AgentRole, Topology


class AgentNetwork:
    """A directed graph of agents with configurable topology."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._agents: dict[str, Agent] = {}

    # ── Construction ─────────────────────────────────────────────────

    def add_agent(self, agent: Agent) -> None:
        self._agents[agent.id] = agent
        self.graph.add_node(agent.id, role=agent.role.value)

    def add_edge(self, src: str, dst: str) -> None:
        self.graph.add_edge(src, dst)

    @property
    def agents(self) -> dict[str, Agent]:
        return self._agents

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agents.keys())

    def neighbors(self, agent_id: str) -> list[str]:
        """Return IDs of agents this agent can send messages to."""
        return list(self.graph.successors(agent_id))

    def predecessors(self, agent_id: str) -> list[str]:
        """Return IDs of agents that can send messages to this agent."""
        return list(self.graph.predecessors(agent_id))

    # ── Topology factories ───────────────────────────────────────────

    @classmethod
    def from_topology(
        cls,
        topology: Topology,
        num_agents: int,
        roles: list[AgentRole] | None = None,
        model: str = "mock",
    ) -> AgentNetwork:
        """Build a network with the given topology."""
        builders = {
            Topology.LINEAR: cls._build_linear,
            Topology.HIERARCHICAL: cls._build_hierarchical,
            Topology.HUB_AND_SPOKE: cls._build_hub_and_spoke,
            Topology.FLAT: cls._build_flat,
            Topology.RING: cls._build_ring,
        }
        net = cls()
        # Create agents
        for i in range(num_agents):
            role = roles[i] if roles and i < len(roles) else AgentRole.GENERIC
            agent = Agent(id=f"agent_{i}", role=role, model=model)
            net.add_agent(agent)
        # Wire them
        builders[topology](net)
        return net

    @staticmethod
    def _build_linear(net: AgentNetwork) -> None:
        """A → B → C → D  (chain)."""
        ids = net.agent_ids
        for i in range(len(ids) - 1):
            net.add_edge(ids[i], ids[i + 1])

    @staticmethod
    def _build_hierarchical(net: AgentNetwork) -> None:
        """Agent 0 is root; each node fans out to up to 2 children."""
        ids = net.agent_ids
        for i, aid in enumerate(ids):
            left = 2 * i + 1
            right = 2 * i + 2
            if left < len(ids):
                net.add_edge(aid, ids[left])
            if right < len(ids):
                net.add_edge(aid, ids[right])

    @staticmethod
    def _build_hub_and_spoke(net: AgentNetwork) -> None:
        """Agent 0 is hub; bidirectional edges to all others."""
        ids = net.agent_ids
        hub = ids[0]
        for spoke in ids[1:]:
            net.add_edge(hub, spoke)
            net.add_edge(spoke, hub)

    @staticmethod
    def _build_flat(net: AgentNetwork) -> None:
        """Fully connected — every agent can message every other."""
        ids = net.agent_ids
        for src in ids:
            for dst in ids:
                if src != dst:
                    net.add_edge(src, dst)

    @staticmethod
    def _build_ring(net: AgentNetwork) -> None:
        """A → B → C → ... → A  (circular)."""
        ids = net.agent_ids
        for i in range(len(ids)):
            net.add_edge(ids[i], ids[(i + 1) % len(ids)])

    # ── Introspection ────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "num_agents": len(self._agents),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
            "diameter": nx.diameter(self.graph.to_undirected())
            if nx.is_weakly_connected(self.graph) else -1,
        }
