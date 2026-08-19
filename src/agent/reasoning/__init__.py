from src.agent.reasoning.base import ReasoningBackend, ReasoningContext
from src.agent.reasoning.rule_based import RuleBasedReasoningBackend

__all__ = ['ReasoningBackend', 'ReasoningContext', 'RuleBasedReasoningBackend', 'get_backend']


def get_backend(name):
    """Resolve a reasoning backend by name. 'rule_based' is always available
    and needs nothing installed. 'ollama' requires a local Ollama server -
    see src/agent/reasoning/ollama_backend.py for what that needs."""
    if name == 'ollama':
        from src.agent.reasoning.ollama_backend import OllamaReasoningBackend
        return OllamaReasoningBackend()
    return RuleBasedReasoningBackend()
