# Black Hat Asia 2027 — Arsenal Submission

## Tool Name

**CogWorm** — Adversarial Propagation Benchmark for Multi-Agent AI Systems

## Short Description

CogWorm is an open-source Python tool that lets security researchers build simulated multi-agent AI networks, inject controlled adversarial test signals, measure propagation across agents, benchmark defenses, and fit mathematical models to predict blast radius — all with a live dashboard and full forensic event logging.

## Detailed Description

### Problem
Multi-agent AI systems (AutoGen, CrewAI, LangGraph) are being deployed in enterprises without any way to measure their exposure to adversarial instruction propagation. When one agent is compromised via prompt injection, how far does the attack spread? Which topology is safest? Which defense actually works? There is currently no tool to answer these questions.

### What CogWorm Does
- **Creates agent networks** with 5 topology types (linear, hierarchical, hub-and-spoke, ring, fully connected)
- **Injects safe synthetic test signals** (static, templated, polymorphic) — no real attacks
- **Traces propagation** with a forensic event logger recording every message, state change, and defense decision
- **Measures metrics**: attack success rate, infection radius, hop survival, signal fidelity decay, defense effectiveness
- **Benchmarks defenses**: 4 built-in defense middlewares (keyword filter, LLM tagger, context isolation, provenance check) as hot-swappable plugins
- **Fits mathematical models**: SIR epidemiological model to predict propagation from topology
- **Visualizes everything**: Streamlit dashboard with network graphs, fidelity decay curves, propagation timelines, and side-by-side defense comparisons

### Demo Plan
1. Show CogWorm CLI: run a baseline experiment in 3 seconds
2. Show the dashboard: configure topology, agents, signal type, defenses
3. Run a live experiment: 10-agent hub-and-spoke, polymorphic signal
4. Show the propagation: animated network graph turning red as agents activate
5. Enable defenses one by one: watch propagation shrink
6. Show the SIR model fit and prediction
7. Show the forensic event log: trace the signal from injection to death
8. Audience participation: let attendees configure and run their own experiments

## Category
- AI/ML Security
- Offensive Tools
- Defensive Tools

## Technical Requirements
- Laptop with Python 3.10+ (self-contained, no cloud needed)
- Power outlet
- Optional: external monitor for dashboard

## Open Source
- License: MIT
- GitHub: [to be published before conference]
- Install: `pip install cogworm`

## Speaker(s)
**[Fill in speaker details]**
