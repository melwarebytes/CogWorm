"""CogWorm Dashboard — interactive experiment visualization.

Run with:  streamlit run cogworm/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from cogworm.core.graph import AgentNetwork
from cogworm.core.logger import EventLogger
from cogworm.core.models import (
    AgentRole,
    AgentState,
    EventType,
    ExperimentConfig,
    SignalType,
    Topology,
)
from cogworm.core.router import Router
from cogworm.core.runner import ExperimentRunner
from cogworm.agents.mock import MockAgent
from cogworm.defenses import DEFENSE_REGISTRY
from cogworm.metrics.calculator import MetricsCalculator
from cogworm.modeling.sir_model import PropagationModel, linear_baseline, exponential_baseline


# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="CogWorm Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #0f3460;
    }
    .stMetric { text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: Experiment Config ───────────────────────────────────────

st.sidebar.title("🧠🪱 CogWorm Lab")
st.sidebar.markdown("---")
st.sidebar.subheader("Experiment Configuration")

topology = st.sidebar.selectbox(
    "Network Topology",
    [t.value for t in Topology],
    format_func=lambda x: x.replace("_", " ").title(),
)

num_agents = st.sidebar.slider("Number of Agents", 3, 20, 5)

signal_type = st.sidebar.selectbox(
    "Signal Type",
    [s.value for s in SignalType],
    format_func=lambda x: x.title(),
)

model_choice = st.sidebar.selectbox(
    "Agent Backend",
    [
        "mock",
        "groq:openai/gpt-oss-120b",
        "groq:openai/gpt-oss-20b",
        "groq:qwen/qwen3.8-27b",
        "groq:allam-2-7b",
        "anthropic:claude-sonnet-4-20250514",
        "openai:gpt-4o",
    ],
    format_func=lambda x: {
        "mock": "🧪 Mock (free, instant)",
        "groq:openai/gpt-oss-120b": "⚡ Groq — GPT-OSS 120B (largest)",
        "groq:openai/gpt-oss-20b": "⚡ Groq — GPT-OSS 20B (fast)",
        "groq:qwen/qwen3.8-27b": "⚡ Groq — Qwen 3.8 27B",
        "groq:allam-2-7b": "⚡ Groq — Allam 2 7B (smallest)",
        "anthropic:claude-sonnet-4-20250514": "🟣 Claude Sonnet 4",
        "openai:gpt-4o": "🟢 GPT-4o",
    }.get(x, x),
)

if model_choice != "mock":
    st.sidebar.caption("⚠️ Requires API key set as env variable")

cycles = st.sidebar.slider("Cycles", 5, 50, 20)
seed = st.sidebar.number_input("Random Seed", 1, 9999, 42)

st.sidebar.markdown("---")
st.sidebar.subheader("Defenses")

use_keyword = st.sidebar.checkbox("Keyword Filter", value=False)
use_llm_tag = st.sidebar.checkbox("LLM Tagger", value=False)
use_isolation = st.sidebar.checkbox("Context Isolation", value=False)
use_provenance = st.sidebar.checkbox("Provenance Check", value=False)

if use_llm_tag:
    detection_acc = st.sidebar.slider("LLM Tagger Accuracy", 0.5, 1.0, 0.85, 0.05)
else:
    detection_acc = 0.85

if use_provenance:
    max_hops = st.sidebar.slider("Max Allowed Hops", 1, 10, 5)
else:
    max_hops = 5

run_button = st.sidebar.button("🚀 Run Experiment", type="primary", use_container_width=True)

compare_button = st.sidebar.button(
    "⚔️ Run Baseline vs Defended", use_container_width=True,
)


# ── Helper functions ─────────────────────────────────────────────────

def build_config(name: str = "dashboard_exp") -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        topology=Topology(topology),
        num_agents=num_agents,
        model=model_choice,
        signal_type=SignalType(signal_type),
        cycles=cycles,
        seed=seed,
    )


def build_defenses() -> list:
    defenses = []
    if use_keyword:
        defenses.append(("keyword_filter", {"type": "keyword_filter"}))
    if use_llm_tag:
        defenses.append(("llm_tagger", {
            "type": "llm_tagger", "mode": "mock",
            "detection_accuracy": detection_acc, "seed": seed,
        }))
    if use_isolation:
        defenses.append(("context_isolation", {"type": "context_isolation"}))
    if use_provenance:
        defenses.append(("provenance_check", {
            "type": "provenance_check", "max_hops": max_hops,
            "require_parent": True,
        }))
    return defenses


def run_single(config: ExperimentConfig, defense_list: list | None = None):
    from cogworm.signals.factory import create_signal as _cs
    network = AgentNetwork.from_topology(
        config.topology, config.num_agents, model=config.model,
    )
    signal_content = _cs(config.signal_type, config.seed)

    # Create backends based on model
    backends = {}
    for aid in network.agent_ids:
        if config.model == "mock":
            backends[aid] = MockAgent(seed=config.seed + hash(aid))
        elif config.model.startswith("groq:"):
            from cogworm.agents.groq_agent import GroqAgent
            backends[aid] = GroqAgent(
                model=config.model.split(":", 1)[1],
                original_signal=signal_content,
            )
        elif config.model.startswith("anthropic:"):
            from cogworm.agents.llm_agent import LLMAgent
            backends[aid] = LLMAgent(
                model=config.model.split(":", 1)[1],
                provider="anthropic",
                original_signal=signal_content,
            )
        elif config.model.startswith("openai:"):
            from cogworm.agents.llm_agent import LLMAgent
            backends[aid] = LLMAgent(
                model=config.model.split(":", 1)[1],
                provider="openai",
                original_signal=signal_content,
            )
        else:
            backends[aid] = MockAgent(seed=config.seed + hash(aid))

    logger = EventLogger()
    router = Router()

    if defense_list:
        for dtype, dcfg in defense_list:
            if dtype in DEFENSE_REGISTRY:
                router.add_defense(DEFENSE_REGISTRY[dtype](config=dcfg))

    runner = ExperimentRunner(config, network, backends, logger, router)
    result = runner.run()
    calc = MetricsCalculator(logger)
    metrics = calc.calculate(result)
    return result, metrics, logger, network


def plot_network(network: AgentNetwork, logger: EventLogger, experiment_id: str):
    """Plot the agent network with propagation state coloring."""
    G = network.graph
    pos = nx.spring_layout(G, seed=42, k=2)

    # Color by state
    state_colors = {
        AgentState.SUSCEPTIBLE: "#4a90d9",
        AgentState.EXPOSED: "#f5a623",
        AgentState.ACTIVATED: "#d0021b",
        AgentState.CONTAINED: "#7ed321",
        AgentState.RECOVERED: "#9b9b9b",
    }

    node_colors = []
    node_text = []
    for aid in G.nodes():
        agent = network.agents[aid]
        node_colors.append(state_colors.get(agent.state, "#4a90d9"))
        node_text.append(
            f"{aid}<br>Role: {agent.role.value}<br>"
            f"State: {agent.state.value}<br>"
            f"Activations: {agent.activation_count}"
        )

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Node positions
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color="#555"),
        hoverinfo="none",
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=35, color=node_colors, line=dict(width=2, color="#fff")),
        text=[aid.replace("agent_", "A") for aid in G.nodes()],
        textposition="middle center",
        textfont=dict(color="white", size=10),
        hovertext=node_text,
        hoverinfo="text",
    ))

    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Legend as annotations
    legend_items = [
        ("Susceptible", "#4a90d9"), ("Exposed", "#f5a623"),
        ("Activated", "#d0021b"), ("Contained", "#7ed321"),
        ("Recovered", "#9b9b9b"),
    ]
    for i, (label, color) in enumerate(legend_items):
        fig.add_annotation(
            x=0.02, y=1.0 - i * 0.08,
            xref="paper", yref="paper",
            text=f"● {label}",
            showarrow=False,
            font=dict(color=color, size=11),
            xanchor="left",
        )

    return fig


def plot_fidelity_by_hop(metrics):
    """Plot signal fidelity decay across hops."""
    if not metrics.fidelity_by_hop:
        return None

    hops = sorted(metrics.fidelity_by_hop.keys())
    fids = [metrics.fidelity_by_hop[h] for h in hops]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hops, y=fids, mode="lines+markers",
        line=dict(color="#d0021b", width=3),
        marker=dict(size=10),
        fill="tozeroy",
        fillcolor="rgba(208,2,27,0.1)",
    ))

    # Threshold line at 0.5
    fig.add_hline(y=0.5, line_dash="dash", line_color="#f5a623",
                  annotation_text="Half-life threshold")
    fig.add_hline(y=0.1, line_dash="dash", line_color="#7ed321",
                  annotation_text="Death threshold")

    fig.update_layout(
        title="Signal Fidelity Decay",
        xaxis_title="Hop Number",
        yaxis_title="Fidelity (0–1)",
        yaxis_range=[0, 1.05],
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_propagation_timeline(logger: EventLogger, experiment_id: str):
    """Plot cumulative propagation over cycles."""
    events = logger.for_experiment(experiment_id)

    cycle_data: dict[int, dict] = {}
    activated = 0
    contained = 0
    messages = 0

    for e in events:
        c = e.cycle
        if c not in cycle_data:
            cycle_data[c] = {"activated": activated, "contained": contained, "messages": messages}
        if e.event_type == EventType.SIGNAL_ACTIVATED:
            activated += 1
        elif e.event_type == EventType.DEFENSE_BLOCKED:
            contained += 1
        elif e.event_type == EventType.MESSAGE_SENT:
            messages += 1
        cycle_data[c] = {"activated": activated, "contained": contained, "messages": messages}

    if not cycle_data:
        return None

    df = pd.DataFrame([
        {"cycle": c, **v} for c, v in sorted(cycle_data.items())
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["activated"], mode="lines+markers",
        name="Activated", line=dict(color="#d0021b", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["contained"], mode="lines+markers",
        name="Contained", line=dict(color="#7ed321", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["messages"], mode="lines",
        name="Messages", line=dict(color="#4a90d9", width=1, dash="dot"),
        yaxis="y2",
    ))

    fig.update_layout(
        title="Propagation Timeline",
        xaxis_title="Cycle",
        yaxis_title="Agent Count",
        yaxis2=dict(title="Messages", overlaying="y", side="right"),
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_sir_fit(result, metrics):
    """Fit SIR model and plot predicted vs observed."""
    if not metrics.fidelity_by_hop:
        return None, None

    # Build time series from cycle data: count activated per cycle
    # For simplicity, use hop as proxy for time step
    hops = sorted(metrics.fidelity_by_hop.keys())
    # Simulate activated count growing with hops
    activated_per_hop = []
    running = 0
    for h in hops:
        if metrics.fidelity_by_hop[h] > 0.1:
            running += 1
        activated_per_hop.append(running)

    if len(hops) < 3:
        return None, None

    model = PropagationModel()
    params = model.fit_sir(hops, activated_per_hop, result.config.num_agents)

    # Predict
    pred = model.predict(result.config.num_agents, max(hops))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hops, y=activated_per_hop, mode="markers",
        name="Observed", marker=dict(size=10, color="#d0021b"),
    ))
    if pred["I"]:
        pred_t = list(range(len(pred["I"])))
        fig.add_trace(go.Scatter(
            x=pred_t, y=pred["I"], mode="lines",
            name=f"SIR Fit (R²={params.r_squared:.3f})",
            line=dict(color="#f5a623", width=2, dash="dash"),
        ))

    # Also fit baselines
    lb = linear_baseline(hops, activated_per_hop)
    fig.add_annotation(
        x=0.98, y=0.02, xref="paper", yref="paper",
        text=(
            f"SIR: β={params.beta:.3f}, γ={params.gamma:.3f}, "
            f"R₀={params.r0:.2f}, R²={params.r_squared:.3f}\n"
            f"Linear R²={lb['r_squared']:.3f}"
        ),
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=10),
        bgcolor="rgba(0,0,0,0.5)",
    )

    fig.update_layout(
        title="SIR Model Fit vs Observed Propagation",
        xaxis_title="Hop / Time Step",
        yaxis_title="Activated Agents",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig, params


# ── Main content ─────────────────────────────────────────────────────

st.title("🧠🪱 CogWorm Lab")
st.caption("Adversarial Propagation Benchmark for Multi-Agent AI Systems")

if run_button:
    config = build_config()
    defense_list = build_defenses()

    with st.spinner("Running experiment..."):
        result, metrics, logger, network = run_single(config, defense_list)

    st.session_state["last_result"] = (result, metrics, logger, network)

    # ── Metrics row ──────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Propagation Rate", f"{result.propagation_rate:.1%}")
    col2.metric("Agents Activated", f"{result.agents_activated}/{config.num_agents}")
    col3.metric("Infection Radius", f"{result.infection_radius} hops")
    col4.metric("Hop Survival", f"{metrics.hop_survival} hops")
    col5.metric("Mean Fidelity", f"{result.mean_signal_fidelity:.3f}")
    col6.metric("Defense Blocks", f"{result.defense_blocks}")

    # ── Visualizations ───────────────────────────────────────────
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Network Graph")
        fig_net = plot_network(network, logger, config.id)
        st.plotly_chart(fig_net, use_container_width=True)

    with right:
        st.subheader("Signal Fidelity Decay")
        fig_fid = plot_fidelity_by_hop(metrics)
        if fig_fid:
            st.plotly_chart(fig_fid, use_container_width=True)
        else:
            st.info("No fidelity data (signal may have been blocked immediately)")

    left2, right2 = st.columns(2)

    with left2:
        fig_timeline = plot_propagation_timeline(logger, config.id)
        if fig_timeline:
            st.plotly_chart(fig_timeline, use_container_width=True)

    with right2:
        fig_sir, sir_params = plot_sir_fit(result, metrics)
        if fig_sir:
            st.plotly_chart(fig_sir, use_container_width=True)
        else:
            st.info("Not enough data points for model fitting")

    # ── Propagation path ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Signal Propagation Path")
    path = logger.propagation_path(config.id)
    if path:
        path_df = pd.DataFrame(path)
        st.dataframe(path_df, use_container_width=True, height=200)
    else:
        st.info("No propagation events recorded")

    # ── XAI Explanation ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧠 XAI — Why Did This Happen?")

    from cogworm.xai.explainer import ExperimentExplainer
    explainer = ExperimentExplainer()
    xai_report = explainer.explain(result, metrics, network, logger)

    # Summary + Risk
    st.markdown(f"**{xai_report.summary}**")
    st.markdown(f"**Risk:** {xai_report.risk_assessment}")

    if xai_report.critical_path:
        st.markdown(f"**Critical Path:** `{' → '.join(xai_report.critical_path)}`")

    # Topology explanation
    with st.expander("🌐 Topology Analysis", expanded=True):
        t = xai_report.topology_explanation
        st.markdown(f"**Risk Level:** `{t.risk_level.upper()}`")
        st.markdown(t.reason)
        for d in t.details:
            st.markdown(f"- {d}")

    # Agent explanations
    with st.expander("🤖 Agent-Level Explanations"):
        for a in xai_report.agent_explanations:
            icon = "🔴" if a.propagated else ("🟡" if a.received_signal else "🔵")
            st.markdown(f"**{icon} {a.agent_id}** ({a.role}) — `{a.final_state}`")
            st.markdown(f"> {a.reason}")
            for d in a.details:
                st.markdown(f"- {d}")
            st.markdown("---")

    # Defense explanations
    with st.expander("🛡️ Defense Analysis"):
        for d in xai_report.defense_explanations:
            st.markdown(f"**{d.defense_id}** — Effectiveness: `{d.effectiveness}`")
            st.markdown(d.reason)
            if d.weaknesses:
                st.markdown("**Weaknesses:**")
                for w in d.weaknesses:
                    st.markdown(f"- ⚠️ {w}")
            if d.recommendations:
                st.markdown("**Recommendations:**")
                for r in d.recommendations:
                    st.markdown(f"- → {r}")

    # Counterfactuals
    with st.expander("🔄 What-If Counterfactuals"):
        for c in xai_report.counterfactuals:
            st.markdown(f"- {c}")

    # Recommendations
    st.markdown("### ✅ Recommendations")
    for r in xai_report.recommendations:
        st.success(r)

    # LLM Deep Analysis (if API key available)
    import os
    if os.environ.get("GROQ_API_KEY"):
        st.markdown("---")
        if st.button("🤖 Generate AI Deep Analysis (Groq)", use_container_width=True):
            with st.spinner("Generating deep analysis with Groq LLM..."):
                from cogworm.xai.llm_explainer import LLMExplainer
                llm_explainer = LLMExplainer()
                deep_analysis = llm_explainer.explain(result, metrics, xai_report)
            st.markdown("### 🤖 AI-Powered Deep Analysis")
            st.markdown(deep_analysis)

    # ── Event log ────────────────────────────────────────────────
    with st.expander("📋 Full Event Log"):
        events = logger.for_experiment(config.id)
        event_data = [{
            "cycle": e.cycle,
            "type": e.event_type.value,
            "source": e.source_agent or "—",
            "dest": e.dest_agent or "—",
            "hop": str(e.hop_number) if e.hop_number is not None else "—",
            "fidelity": f"{e.signal_fidelity:.3f}" if e.signal_fidelity is not None else "—",
            "defense": e.defense_id or "—",
        } for e in events if e.event_type not in (EventType.CYCLE_START, EventType.CYCLE_END)]
        if event_data:
            st.dataframe(pd.DataFrame(event_data), use_container_width=True, height=400)

elif compare_button:
    # ── Comparison mode ──────────────────────────────────────────
    config = build_config("comparison")
    defense_list = build_defenses()

    if not defense_list:
        st.warning("Enable at least one defense to compare against baseline!")
    else:
        col_b, col_d = st.columns(2)

        with st.spinner("Running baseline (no defenses)..."):
            r_base, m_base, l_base, n_base = run_single(config)

        with st.spinner("Running defended..."):
            r_def, m_def, l_def, n_def = run_single(config, defense_list)

        calc = MetricsCalculator(l_base)
        comparison = calc.compare(m_base, m_def)

        st.markdown("---")
        st.subheader("⚔️ Baseline vs Defended Comparison")

        # Side-by-side metrics
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Propagation Rate",
            f"{r_def.propagation_rate:.1%}",
            delta=f"{r_def.propagation_rate - r_base.propagation_rate:+.1%}",
            delta_color="inverse",
        )
        c2.metric(
            "Agents Activated",
            f"{r_def.agents_activated}",
            delta=f"{r_def.agents_activated - r_base.agents_activated:+d}",
            delta_color="inverse",
        )
        c3.metric(
            "Defense Effectiveness",
            f"{comparison['defense_effectiveness']:.1%}",
        )

        # Side-by-side networks
        left, right = st.columns(2)
        with left:
            st.markdown("**Baseline (No Defenses)**")
            st.plotly_chart(
                plot_network(n_base, l_base, config.id),
                use_container_width=True,
            )
            st.caption(
                f"Propagation: {r_base.propagation_rate:.1%} | "
                f"Activated: {r_base.agents_activated}/{config.num_agents} | "
                f"Radius: {r_base.infection_radius} hops"
            )

        with right:
            defense_names = [d[0].replace("_", " ").title() for d in defense_list]
            st.markdown(f"**Defended ({', '.join(defense_names)})**")
            st.plotly_chart(
                plot_network(n_def, l_def, config.id),
                use_container_width=True,
            )
            st.caption(
                f"Propagation: {r_def.propagation_rate:.1%} | "
                f"Activated: {r_def.agents_activated}/{config.num_agents} | "
                f"Blocks: {r_def.defense_blocks}"
            )

        # Comparison bar chart
        st.markdown("---")
        comp_df = pd.DataFrame([
            {"Metric": "Propagation Rate", "Baseline": r_base.propagation_rate, "Defended": r_def.propagation_rate},
            {"Metric": "Mean Fidelity", "Baseline": r_base.mean_signal_fidelity, "Defended": r_def.mean_signal_fidelity},
            {"Metric": "Infection Radius (norm)", "Baseline": r_base.infection_radius / max(config.num_agents, 1), "Defended": r_def.infection_radius / max(config.num_agents, 1)},
        ])
        fig_comp = px.bar(
            comp_df.melt(id_vars="Metric", var_name="Condition", value_name="Value"),
            x="Metric", y="Value", color="Condition",
            barmode="group",
            color_discrete_map={"Baseline": "#d0021b", "Defended": "#7ed321"},
            title="Defense Effectiveness",
        )
        fig_comp.update_layout(height=350)
        st.plotly_chart(fig_comp, use_container_width=True)

else:
    # ── Landing page ─────────────────────────────────────────────
    st.markdown("""
    ### Welcome to CogWorm Lab

    **CogWorm** is a controlled benchmark and simulation platform for studying
    how adversarial instructions propagate through multi-agent AI systems.

    **Getting started:**
    1. Configure your experiment in the sidebar
    2. Choose a topology, signal type, and optional defenses
    3. Click **🚀 Run Experiment** to see propagation results
    4. Click **⚔️ Run Baseline vs Defended** to compare defense effectiveness

    **What you'll see:**
    - 🔴 Network graph showing which agents got infected
    - 📉 Signal fidelity decay across hops
    - 📊 Propagation timeline across cycles
    - 🧮 SIR epidemiological model fit
    - 📋 Full forensic event log with message lineage
    """)

    # Show all topologies as a preview
    st.subheader("Available Topologies")
    cols = st.columns(5)
    for i, topo in enumerate(Topology):
        with cols[i]:
            net = AgentNetwork.from_topology(topo, 6)
            G = net.graph
            pos = nx.spring_layout(G, seed=42, k=2)
            fig = go.Figure()
            for u, v in G.edges():
                fig.add_trace(go.Scatter(
                    x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]],
                    mode="lines", line=dict(width=1, color="#555"),
                    hoverinfo="none", showlegend=False,
                ))
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in G.nodes()],
                y=[pos[n][1] for n in G.nodes()],
                mode="markers",
                marker=dict(size=15, color="#4a90d9"),
                hoverinfo="none", showlegend=False,
            ))
            fig.update_layout(
                title=topo.value.replace("_", " ").title(),
                height=200, margin=dict(l=5, r=5, t=30, b=5),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
