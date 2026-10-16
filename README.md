# CogWorm 🧠🪱

**A controlled benchmark and simulation platform for studying adversarial instruction propagation in multi-agent AI systems.**

CogWorm is a research tool — a laboratory for measuring and modelling how adversarial instructions ("cognitive worms") propagate through networks of AI agents, and how well defenses contain them.

## What CogWorm Does

1. **Build** a simulated multi-agent network (linear, hierarchical, hub-and-spoke, ring, or fully connected)
2. **Inject** a controlled synthetic test signal (static, templated, or polymorphic)
3. **Observe** how the signal propagates across agents with full message lineage tracking
4. **Measure** propagation rate, infection radius, signal fidelity decay, and hop survival
5. **Test** defense middlewares (keyword filter, LLM tagger, context isolation, provenance checks)
6. **Model** propagation mathematically (SIR epidemiological model vs baselines)
7. **Compare** topologies and defenses with reproducible, config-driven experiments

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run a baseline experiment
cogworm run configs/baseline_linear.yaml

# Run with defenses
cogworm run configs/defended_keyword.yaml

# Run tests
pytest tests/ -v
```

## Project Structure

```
cogworm/
├── cogworm/
│   ├── core/           # Data models, graph, router, runner, event logger
│   ├── agents/         # Agent backends (mock + LLM-powered)
│   ├── signals/        # Synthetic test signal generation
│   ├── defenses/       # Defense middleware plugins
│   ├── metrics/        # Propagation & defense metrics
│   ├── modeling/       # SIR model + baselines
│   ├── dashboard/      # Streamlit visualization (Phase 5)
│   └── cli.py          # Command-line interface
├── configs/            # Experiment configurations (YAML)
├── datasets/           # Experiment output datasets
├── tests/              # Test suite
└── docs/               # Documentation
```

## Architecture

CogWorm has two layers:

**Layer 1 — CogWorm Lab** (the experimental platform):
- Agent graphs with 5 topology types
- Framework-independent agent abstraction (mock or real LLM)
- Message routing with defense middleware hooks
- Complete event logging with forensic lineage tracking
- Config-driven reproducible experiments

**Layer 2 — CogWorm Model** (the analytics layer):
- Propagation metrics (ASR, infection radius, hop survival, fidelity)
- Defense effectiveness comparison
- SIR epidemiological model fitting
- Baseline comparisons (linear, exponential)
- Prediction on unseen configurations

## Defense Middleware

Every defense implements:
```python
before_send(message, context) -> allow | block | transform
before_receive(message, context) -> allow | block | transform
```

Built-in defenses:
- **Keyword Filter** — pattern-based detection
- **LLM Tagger** — ML classification of adversarial content  
- **Context Isolation** — role-based content stripping
- **Provenance Check** — lineage validation and hop limits

## Research Questions

- **RQ1 (Topology):** How does propagation change across network structures?
- **RQ2 (Semantic Decay):** How quickly does signal fidelity degrade per hop?
- **RQ3 (Polymorphism):** Can signals preserve effect while changing wording?
- **RQ4 (Formal Model):** Can propagation be predicted from topology + model properties?
- **RQ5 (Defenses):** Which containment strategies are most effective?

## License

MIT
