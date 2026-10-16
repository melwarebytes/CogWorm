"""CogWorm XAI — Explainable AI layer.

Answers the "why" questions:
- Why did the signal propagate through this agent?
- Why did this topology amplify/suppress propagation?
- Why did this defense work (or fail)?
- What was the critical path?
- What would happen if we changed X?
"""

from cogworm.xai.explainer import ExperimentExplainer
from cogworm.xai.llm_explainer import LLMExplainer
