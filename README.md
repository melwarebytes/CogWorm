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
7. **Explain** results with rule-based and LLM-powered XAI
8. **Compare** topologies and defenses with reproducible, config-driven experiments

## Installation

### Prerequisites
- Python 3.10+
- (Optional) A [Groq API key](https://console.groq.com/) for real LLM experiments

### Setup

```bash
# Clone the repository
git clone https://github.com/melwarebytes/CogWorm.git
cd CogWorm

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# (Optional) Set Groq API key for real LLM experiments
export GROQ_API_KEY="your_groq_api_key_here"
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Run a quick mock experiment
cogworm run configs/baseline_linear.yaml
```

## Usage

### 1. CLI — Run Experiments

```bash
# Run a baseline experiment (mock agents, instant)
cogworm run configs/baseline_linear.yaml

# Run with all defenses enabled
cogworm run configs/defended_all.yaml

# Run with real Groq LLM (needs GROQ_API_KEY)
cogworm run configs/groq_hub_gptoss120b.yaml

# Run polymorphic worm on flat topology (worst case)
cogworm run configs/groq_flat_polymorphic.yaml

# Save results to custom directory
cogworm run configs/baseline_linear.yaml --output results/my_experiment
```

### 2. Dashboard — Interactive Visualization

```bash
# Launch the Streamlit dashboard
streamlit run cogworm/dashboard/app.py
```

Then open `http://localhost:8501` in your browser. The dashboard lets you:

- Pick topology, signal type, agent backend, and defenses from the sidebar
- Click **"Run Experiment"** to execute and visualize results
- Click **"Run Baseline vs Defended"** for side-by-side comparison
- View network graph, fidelity decay, propagation timeline, SIR model fit
- Expand XAI sections for per-agent, topology, and defense explanations
- Click **"Generate AI Deep Analysis"** for LLM-powered explanation (needs Groq key)

### 3. Custom Experiment Configs

Create a YAML file in `configs/`:

```yaml
experiment:
  name: my_experiment
  topology: hub_and_spoke       # linear | hierarchical | hub_and_spoke | ring | flat
  agents: 5
  agent_roles:
    - orchestrator
    - doc_retrieval
    - email_drafting
    - database_query
    - code_execution
  model: mock                   # mock | groq:openai/gpt-oss-20b | groq:openai/gpt-oss-120b
  signal_type: static           # static | templated | polymorphic
  cycles: 20
  seed: 42
  defenses:
    - id: keyword_filter
      type: keyword_filter
      config:
        type: keyword_filter
    - id: provenance_check
      type: provenance_check
      config:
        type: provenance_check
        max_hops: 3
```

Then run: `cogworm run configs/my_experiment.yaml`

## Project Structure

```
CogWorm/
├── cogworm/
│   ├── core/               # Data models, graph, router, runner, event logger
│   │   ├── models.py       # Pydantic models (Agent, Message, Event, Config)
│   │   ├── graph.py        # AgentNetwork with 5 topology factories
│   │   ├── router.py       # Message router with defense middleware hooks
│   │   ├── runner.py       # ExperimentRunner — the simulation engine
│   │   └── logger.py       # SQLite-backed event logger with lineage tracking
│   ├── agents/             # Agent backends
│   │   ├── mock.py         # Deterministic mock agent (zero-cost testing)
│   │   ├── groq_agent.py   # Groq LLM agent with three-way fidelity
│   │   └── llm_agent.py    # OpenAI/Anthropic LLM agent
│   ├── signals/            # Synthetic test signal generation
│   │   └── factory.py      # Static, templated, polymorphic signal factory
│   ├── defenses/           # Defense middleware plugins
│   │   ├── keyword_filter.py    # Regex pattern matching
│   │   ├── llm_tagger.py       # ML classification
│   │   ├── context_isolation.py # Role-based content stripping
│   │   └── provenance_check.py  # Lineage validation + hop limits
│   ├── metrics/            # Propagation & defense metrics calculator
│   ├── modeling/           # SIR epidemiological model + baselines
│   ├── xai/                # Explainable AI layer
│   │   ├── explainer.py    # Rule-based explanations + counterfactuals
│   │   └── llm_explainer.py # LLM-powered deep analysis (via Groq)
│   ├── dashboard/          # Streamlit interactive dashboard
│   └── cli.py              # Command-line interface
├── configs/                # 13 experiment configurations (YAML)
├── datasets/               # Experiment output datasets
├── tests/                  # 12-test suite
├── docs/                   # Black Hat Asia 2027 proposals
├── requirements.txt
└── pyproject.toml
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
- XAI explanations (rule-based + LLM-powered)

## Defense Middleware

Every defense implements:
```python
before_send(message, context) -> allow | block | transform
before_receive(message, context) -> allow | block | transform
```

Built-in defenses:
- **Keyword Filter** — pattern-based detection of adversarial instructions
- **LLM Tagger** — ML classification of adversarial content  
- **Context Isolation** — role-based content stripping for non-orchestrator agents
- **Provenance Check** — lineage validation, hop limits, and orphan message blocking

## Available Groq Models

| Model | ID | Notes |
|---|---|---|
| GPT-OSS 120B | `groq:openai/gpt-oss-120b` | Largest, most susceptible to propagation |
| GPT-OSS 20B | `groq:openai/gpt-oss-20b` | Medium, good balance of speed and capability |
| Qwen 3.8 27B | `groq:qwen/qwen3.8-27b` | Resistant — actively refuses adversarial signals |
| Allam 2 7B | `groq:allam-2-7b` | Smallest available model |

## Research Questions

- **RQ1 (Topology):** How does propagation change across network structures?
- **RQ2 (Semantic Decay):** How quickly does signal fidelity degrade per hop?
- **RQ3 (Polymorphism):** Can signals preserve effect while changing wording?
- **RQ4 (Formal Model):** Can propagation be predicted from topology + model properties?
- **RQ5 (Defenses):** Which containment strategies are most effective?

## Key Findings (Preliminary)

| Config | Topology | Signal | Propagation | Hops | Fidelity |
|---|---|---|---|---|---|
| GPT-OSS-20B | Linear | Static | 25% | 1 | 0.816 |
| GPT-OSS-120B | Hub-Spoke | Static | 0% | 3 | 0.697 |
| GPT-OSS-20B | Flat | Static | 25% | 2 | 0.694 |
| GPT-OSS-20B | Flat | Polymorphic | **50%** | **21** | 0.223 |

The polymorphic signal on flat topology achieves the highest propagation with persistent re-infection loops between agents.

## License

MIT
