"""LLM-powered explainer — uses Groq to generate deep explanations.

Takes the structured XAI report and the raw experiment data, and uses
an LLM to generate a natural-language deep-dive explanation that a
security analyst or researcher can act on.
"""

from __future__ import annotations

import json
import os
from typing import Any

from cogworm.core.models import ExperimentResult
from cogworm.metrics.calculator import PropagationMetrics
from cogworm.xai.explainer import ExperimentExplanationReport


ANALYSIS_PROMPT = """\
You are CogWorm's XAI analyst. You explain adversarial signal propagation
experiments in multi-agent AI systems.

Given the experiment results below, provide a clear, actionable analysis that
a security researcher or red/blue team member can use. Structure your response
as:

1. **Executive Summary** (2-3 sentences: what happened, how bad is it)
2. **Why It Propagated** (or didn't): explain the causal chain
3. **Critical Vulnerabilities**: which agents/edges were the weak links
4. **Topology Impact**: how the network structure amplified or suppressed spread
5. **Defense Assessment**: what worked, what didn't, why
6. **Attack Surface Analysis**: what an attacker would learn from this
7. **Recommended Mitigations**: specific, prioritised actions

Be direct and technical. Use the actual numbers from the data.
"""


class LLMExplainer:
    """Generate natural-language experiment explanations using an LLM."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        provider: str = "groq",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.provider == "groq":
            if not self._api_key:
                raise ValueError("GROQ_API_KEY not set")
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
            except ImportError:
                import openai
                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
        return self._client

    def explain(
        self,
        result: ExperimentResult,
        metrics: PropagationMetrics,
        report: ExperimentExplanationReport,
    ) -> str:
        """Generate a deep natural-language explanation using the LLM."""
        client = self._get_client()

        # Build the data context for the LLM
        data_context = self._build_context(result, metrics, report)

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.3,  # lower temp for analytical output
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": data_context},
                ],
            )
            return response.choices[0].message.content or "Analysis unavailable."
        except Exception as e:
            return f"LLM analysis unavailable: {e}"

    def _build_context(
        self,
        result: ExperimentResult,
        metrics: PropagationMetrics,
        report: ExperimentExplanationReport,
    ) -> str:
        """Build a structured data context for the LLM to analyse."""
        config = result.config

        context = f"""## Experiment Data

**Configuration:**
- Topology: {config.topology.value}
- Agents: {config.num_agents}
- Model: {config.model}
- Signal type: {config.signal_type.value}
- Cycles: {config.cycles}
- Defenses active: {len(config.defenses)}

**Results:**
- Propagation rate: {result.propagation_rate:.1%}
- Agents activated: {result.agents_activated}/{config.num_agents}
- Agents contained: {result.agents_contained}
- Infection radius: {result.infection_radius} hops
- Hop survival: {metrics.hop_survival} hops
- Mean signal fidelity: {result.mean_signal_fidelity:.3f}
- Defense blocks: {result.defense_blocks}
- Total messages: {result.total_messages}
- Duration: {result.duration_seconds:.2f}s

**Risk Assessment:** {report.risk_assessment}

**Critical Path:** {' → '.join(report.critical_path) if report.critical_path else 'None'}

**Fidelity by Hop:** {json.dumps(metrics.fidelity_by_hop, indent=2) if metrics.fidelity_by_hop else 'N/A'}

**Agent States:**
"""
        for a in report.agent_explanations:
            context += f"- {a.agent_id} ({a.role}): {a.final_state}"
            if a.fidelity_at_receipt is not None:
                context += f" | fidelity={a.fidelity_at_receipt:.3f}"
            if a.blocked_by:
                context += f" | blocked by {a.blocked_by}"
            context += "\n"

        context += f"\n**Topology Analysis:** {report.topology_explanation.reason}\n"
        context += f"Risk level: {report.topology_explanation.risk_level}\n"

        if report.defense_explanations:
            context += "\n**Defense Performance:**\n"
            for d in report.defense_explanations:
                context += f"- {d.defense_id}: {d.blocks} blocks, {d.bypasses} bypasses — {d.effectiveness}\n"

        return context
