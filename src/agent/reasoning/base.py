"""The swap point between rule-based and LLM-backed reasoning.

Both RuleBasedReasoningBackend and OllamaReasoningBackend implement this same
interface and are handed the same ReasoningContext - a snapshot of state plus
a logging callback and the same tool registry. The rest of the agent (core.py,
the AI Agent page, memory) does not know or care which one produced a given
recommendation.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ReasoningContext:
    """Everything a reasoning backend needs for one cycle."""
    state: dict
    company_id: str
    cycle_id: str
    log: callable  # log(step, message, signal_type=None, data=None) -> None
    signals_found: int = field(default=0)
    actions_taken: int = field(default=0)

    def note_signal(self):
        self.signals_found += 1

    def note_action(self):
        self.actions_taken += 1


class ReasoningBackend(ABC):
    """REASON + PLAN + ACT for one cycle, given a perceived state.

    PERCEIVE (building the state snapshot) and OBSERVE (checking whether past
    recommendations played out) happen in src/agent/core.py, identically
    regardless of backend - only the "what's significant, and what should we
    do about it" step varies.
    """

    name = 'base'

    @abstractmethod
    def run_cycle(self, ctx: ReasoningContext) -> None:
        """Evaluate signals in `ctx.state`, and for each significant one, plan
        and execute a chain of tool calls (segment, generate a plan, log a
        recommendation, raise an alert). Call `ctx.log(...)` at each step so
        the trail is legible, and `ctx.note_signal()` / `ctx.note_action()`
        so the cycle summary is accurate."""
        raise NotImplementedError
