"""Core tests — verify the full pipeline works end-to-end."""

import pytest
from cogworm.core.models import (
    AgentRole, ExperimentConfig, SignalType, Topology,
)
from cogworm.core.graph import AgentNetwork
from cogworm.core.logger import EventLogger
from cogworm.core.router import Router
from cogworm.core.runner import ExperimentRunner
from cogworm.agents.mock import MockAgent
from cogworm.metrics.calculator import MetricsCalculator
from cogworm.signals.factory import create_signal, detect_marker


class TestAgentNetwork:
    def test_linear_topology(self):
        net = AgentNetwork.from_topology(Topology.LINEAR, 5)
        assert len(net.agent_ids) == 5
        assert net.neighbors("agent_0") == ["agent_1"]
        assert net.neighbors("agent_4") == []

    def test_hierarchical_topology(self):
        net = AgentNetwork.from_topology(Topology.HIERARCHICAL, 7)
        assert set(net.neighbors("agent_0")) == {"agent_1", "agent_2"}

    def test_hub_and_spoke(self):
        net = AgentNetwork.from_topology(Topology.HUB_AND_SPOKE, 5)
        assert len(net.neighbors("agent_0")) == 4  # hub → all spokes

    def test_ring_topology(self):
        net = AgentNetwork.from_topology(Topology.RING, 4)
        assert "agent_1" in net.neighbors("agent_0")
        assert "agent_0" in net.neighbors("agent_3")  # wraps around

    def test_flat_topology(self):
        net = AgentNetwork.from_topology(Topology.FLAT, 3)
        assert len(net.neighbors("agent_0")) == 2  # connected to all others


class TestSignals:
    def test_static_signal(self):
        signal = create_signal(SignalType.STATIC, seed=42)
        assert "COGWORM_MARKER_" in signal
        assert detect_marker(signal, seed=42)

    def test_templated_signal(self):
        signal = create_signal(SignalType.TEMPLATED, seed=42)
        assert "COGWORM_MARKER_" in signal

    def test_polymorphic_signal(self):
        signal = create_signal(SignalType.POLYMORPHIC, seed=42)
        assert "rephrase" in signal.lower()

    def test_different_seeds_different_markers(self):
        s1 = create_signal(SignalType.STATIC, seed=1)
        s2 = create_signal(SignalType.STATIC, seed=2)
        assert s1 != s2


class TestExperimentRunner:
    def test_end_to_end_no_defense(self):
        config = ExperimentConfig(
            name="test_e2e",
            topology=Topology.LINEAR,
            num_agents=3,
            signal_type=SignalType.STATIC,
            cycles=10,
            seed=42,
        )
        network = AgentNetwork.from_topology(config.topology, config.num_agents)
        backends = {aid: MockAgent(seed=42) for aid in network.agent_ids}
        logger = EventLogger()
        runner = ExperimentRunner(config, network, backends, logger)

        result = runner.run()

        assert result.total_messages > 0
        assert result.total_events > 0
        assert result.propagation_rate >= 0
        assert result.infection_radius >= 0

    def test_defense_reduces_propagation(self):
        from cogworm.defenses.keyword_filter import KeywordFilterDefense

        config = ExperimentConfig(
            name="test_defended",
            topology=Topology.LINEAR,
            num_agents=5,
            signal_type=SignalType.STATIC,
            cycles=10,
            seed=42,
        )

        # Baseline (no defense)
        net1 = AgentNetwork.from_topology(config.topology, config.num_agents)
        backends1 = {aid: MockAgent(seed=42) for aid in net1.agent_ids}
        logger1 = EventLogger()
        r1 = ExperimentRunner(config, net1, backends1, logger1).run()

        # With keyword filter
        net2 = AgentNetwork.from_topology(config.topology, config.num_agents)
        backends2 = {aid: MockAgent(seed=42) for aid in net2.agent_ids}
        logger2 = EventLogger()
        router = Router()
        router.add_defense(KeywordFilterDefense())
        r2 = ExperimentRunner(config, net2, backends2, logger2, router).run()

        # Defense should block or reduce propagation
        assert r2.propagation_rate <= r1.propagation_rate or r2.defense_blocks > 0


class TestMetrics:
    def test_metrics_from_experiment(self):
        config = ExperimentConfig(
            name="test_metrics",
            topology=Topology.HUB_AND_SPOKE,
            num_agents=5,
            cycles=10,
            seed=42,
        )
        network = AgentNetwork.from_topology(config.topology, config.num_agents)
        backends = {aid: MockAgent(seed=42) for aid in network.agent_ids}
        logger = EventLogger()
        result = ExperimentRunner(config, network, backends, logger).run()

        calc = MetricsCalculator(logger)
        metrics = calc.calculate(result)

        assert metrics.agents_total == 5
        assert 0 <= metrics.attack_success_rate <= 1
        assert metrics.infection_radius >= 0
