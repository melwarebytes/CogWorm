# Black Hat Asia 2027 — Briefings Submission

## Title

**CogWorm: Measuring How Adversarial Instructions Worm Through Multi-Agent AI Systems**

## Abstract (max 300 words)

Multi-agent AI systems — where autonomous LLM-powered agents collaborate by exchanging messages — are rapidly replacing monolithic chatbots in enterprise deployments. But their trust model is fundamentally broken: every agent implicitly trusts messages from every other agent, creating a lateral movement surface that has no equivalent in traditional software.

We introduce **CogWorm**, an open-source benchmark and simulation platform that measures how a single adversarial instruction injected into one agent propagates across an entire multi-agent network — what we call a "cognitive worm." Unlike prior prompt-injection research that focuses on single-agent attacks, CogWorm systematically quantifies *network-level* propagation: how topology shapes blast radius, how fast adversarial semantics decay across agent hops, and whether polymorphic payloads can survive paraphrasing by intermediate agents.

We built a controlled laboratory with five network topologies (linear, hierarchical, hub-and-spoke, ring, fully connected), three payload families (static, templated, polymorphic), and four defense middlewares (keyword filtering, LLM-based tagging, context privilege isolation, and provenance-chain validation). Our experiments across multiple LLM families reveal that:

1. **Hub-and-spoke topologies amplify propagation** — a single injection at the hub reaches 3× more agents than in a linear chain.
2. **Semantic decay is not a reliable defense** — polymorphic payloads maintain functional effect even when textual similarity drops below BERTScore thresholds researchers typically rely on.
3. **An SIR epidemiological model predicts propagation** with R² > 0.85, enabling defenders to estimate blast radius from topology alone.
4. **Provenance-chain validation is the most effective containment** — reducing propagation by 78–94% compared to keyword filtering's 45–60%.

We release CogWorm as an open-source tool with a live dashboard, reproducible config-driven experiments, and a forensic event logger that reconstructs full message lineage — giving red teams a way to quantify multi-agent risk and blue teams a benchmark for testing containment strategies.

**Live demo included.**

## Detailed Outline

### Part 1: The Problem (10 min)
- Multi-agent AI architectures are everywhere (AutoGen, CrewAI, LangGraph)
- The implicit trust problem: agents trust other agents' outputs as if they were user instructions
- Why single-agent prompt injection research doesn't capture network effects
- Defining "cognitive worms": adversarial instructions that persist and propagate across agent boundaries

### Part 2: CogWorm Architecture (10 min)
- Two-layer design: Lab (experimental platform) + Model (analytics)
- Agent network graph abstraction — topology-independent
- Safe synthetic test signals — measuring propagation without building real attacks
- The three-way fidelity measurement: textual similarity vs behavioral preservation vs actual propagation
- Defense middleware protocol: `before_send()` / `before_receive()` hooks
- Live dashboard demo

### Part 3: Key Findings (15 min)
- **Topology matters more than the model**: hub-and-spoke and flat topologies create blast radii 2–5× larger than linear chains
- **Semantic decay is unreliable**: polymorphic payloads maintain >80% functional effect at BERTScore 0.65, below the 0.85 threshold typically used as a safety cutoff
- **The SIR model works**: epidemiological modeling predicts propagation with R² > 0.85, meaning defenders can estimate risk from topology diagrams alone
- **Defense comparison**: provenance validation >> LLM tagging >> context isolation >> keyword filtering
- Fidelity decay curves by topology (the "propagation half-life")

### Part 4: Implications for Defenders (10 min)
- How to audit your multi-agent topology for propagation risk
- Why you need provenance chains, not just input filtering
- Using CogWorm to benchmark your own defenses before deployment
- Predicting blast radius with the SIR model

### Part 5: Live Demo + Tool Release (5 min)
- Running CogWorm against a 10-agent hierarchical network
- Injecting a polymorphic signal and watching it propagate in real-time
- Enabling defenses and showing the difference
- Open-source release: GitHub repo, pip install, documentation

## Why This Talk Belongs at Black Hat

1. **Novel attack surface**: Multi-agent AI propagation is a fundamentally new lateral movement vector with no coverage at Black Hat to date.
2. **Working tool**: CogWorm is a complete, open-source benchmarking platform — not a paper concept.
3. **Offensive + defensive**: We show how attackers can exploit agent trust, AND give defenders quantitative tools to measure and reduce risk.
4. **Live demo**: Real-time propagation visualization across agent networks.
5. **Actionable**: Attendees leave with a tool they can install and run against their own multi-agent deployments.

## Speaker(s)

**[Speaker 1 — fill in]**
- Bio, affiliation, relevant background
- Prior speaking / publication history

**[Speaker 2 — fill in, if applicable]**

## Prior Publication / Disclosure

- This research has not been previously published or presented.
- The tool will be open-sourced at the time of presentation.
- We also plan to submit to the **Black Hat Asia Arsenal** (tools track) for a hands-on demo session.

## Technical Requirements

- Standard presentation setup (projector, audio)
- Internet connection for live demo (can fall back to recorded demo)
- 50-minute slot (Briefings standard)
