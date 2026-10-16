"""Defense middleware plugins.

Each defense implements the DefenseMiddleware protocol:
    before_send(message, context) -> allow/block/transform
    before_receive(message, context) -> allow/block/transform

Defenses are hot-swappable via configuration — no code changes needed.
"""

from cogworm.defenses.keyword_filter import KeywordFilterDefense
from cogworm.defenses.llm_tagger import LLMTaggerDefense
from cogworm.defenses.context_isolation import ContextIsolationDefense
from cogworm.defenses.provenance_check import ProvenanceCheckDefense

DEFENSE_REGISTRY: dict[str, type] = {
    "keyword_filter": KeywordFilterDefense,
    "llm_tagger": LLMTaggerDefense,
    "context_isolation": ContextIsolationDefense,
    "provenance_check": ProvenanceCheckDefense,
}
