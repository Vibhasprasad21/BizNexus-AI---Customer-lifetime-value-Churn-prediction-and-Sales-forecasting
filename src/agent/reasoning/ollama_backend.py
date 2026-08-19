"""Optional LLM-backed reasoning backend, using a local Ollama server.

THIS IS OFF BY DEFAULT. See src/agent/config.py - REASONING_BACKEND stays
'rule_based' unless BIZNEXUS_AGENT_BACKEND=ollama is set in the environment.

What this buys you over RuleBasedReasoningBackend: it can reason about
situations nobody wrote an explicit rule for - "the churn spike and the
revenue dip both trace back to the same three enterprise accounts, treat
them as one issue" is the kind of judgment call a fixed threshold list can't
make but a model reading the same tool outputs can. What it costs: Ollama
has to be installed and running (`ollama serve`, with a model pulled, e.g.
`ollama pull llama3`) on whatever machine runs this app. That's true for a
self-hosted or local deployment, and NOT true for Streamlit Community Cloud,
which only runs the app itself - there's nowhere for Ollama to run alongside
it. On Community Cloud this backend will fail its connectivity check and the
agent should stay on 'rule_based'.

Implementation: a small bounded ReAct loop. The model sees the same tool
docstrings the rule engine has access to (src/agent/tools.TOOL_DOCS) and a
summary of the perceived state, and responds turn by turn with either a tool
call or a final decision. Every tool call and its result is logged to the
same reasoning trail as the rule-based backend, so the Agent Activity Log
looks the same regardless of which backend produced it.
"""
import json
import re

import requests

from src.agent import config, tools
from src.agent.reasoning.base import ReasoningBackend, ReasoningContext

MAX_ITERATIONS = 6
REQUEST_TIMEOUT = 60

SYSTEM_PROMPT = """You are the reasoning engine for a business analytics agent.
You are given read-only tools to inspect a company's CLV, churn, and sales
forecast data, and action tools to raise alerts and log recommendations.

Available tools:
{tool_docs}

On each turn, respond with exactly one of:
CALL: <tool_name>(<json object of kwargs>)
DONE: <one-paragraph summary of what you found and what you did>

Only call action tools (create_alert, log_recommendation) for signals you can
justify with numbers you actually pulled via the read-only tools this turn or
earlier. Stop once you've investigated the state and logged whatever
recommendations are warranted - usually within 3-5 tool calls. Do not repeat
an identical call twice."""


class OllamaReasoningBackend(ReasoningBackend):
    name = 'ollama'

    def __init__(self, host=None, model=None):
        self.host = host or config.OLLAMA_HOST
        self.model = model or config.OLLAMA_MODEL

    def is_available(self):
        """Connectivity check - call before use. False almost certainly means
        Ollama isn't installed/running here (e.g. Streamlit Community Cloud)."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def run_cycle(self, ctx: ReasoningContext) -> None:
        if not self.is_available():
            ctx.log('reason', f"Ollama backend unavailable at {self.host} - "
                               f"is `ollama serve` running with model '{self.model}' pulled? "
                               f"Falling back to no action this cycle.")
            return

        tool_docs = "\n".join(f"- {name}: {doc}" for name, doc in tools.TOOL_DOCS.items())
        state_summary = self._summarize_state(ctx.state)
        transcript = [
            {'role': 'system', 'content': SYSTEM_PROMPT.format(tool_docs=tool_docs)},
            {'role': 'user', 'content': f"Current perceived state:\n{state_summary}\n\n"
                                         f"Company ID for any tool calls: {ctx.company_id}"},
        ]

        for _ in range(MAX_ITERATIONS):
            reply = self._chat(transcript)
            if reply is None:
                ctx.log('reason', "Ollama call failed or timed out; stopping this cycle early.")
                return

            transcript.append({'role': 'assistant', 'content': reply})
            call = self._parse_call(reply)

            if call is None:
                summary = reply.split('DONE:', 1)[-1].strip() if 'DONE:' in reply else reply.strip()
                ctx.log('observe', summary or "Model finished without a summary.")
                return

            name, kwargs = call
            func = tools.TOOL_REGISTRY.get(name)
            if not func:
                observation = f"Error: unknown tool '{name}'."
                transcript.append({'role': 'user', 'content': observation})
                continue

            kwargs = self._bind_context(name, kwargs, ctx)
            step = 'act' if name in ('create_alert', 'log_recommendation') else 'reason'
            try:
                result = func(**kwargs)
            except Exception as e:
                result = {'error': str(e)}

            if step == 'act':
                ctx.note_action()
            ctx.log(step, f"Model called {name}({kwargs}) -> {self._truncate(result)}", data=result)
            transcript.append({'role': 'user', 'content': f"Result: {json.dumps(result, default=str)[:2000]}"})

        ctx.log('observe', f"Stopped after {MAX_ITERATIONS} tool calls without a final DONE.")

    # --- helpers -------------------------------------------------------------

    def _chat(self, transcript):
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={'model': self.model, 'messages': transcript, 'stream': False},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get('message', {}).get('content', '').strip()
        except Exception:
            return None

    @staticmethod
    def _parse_call(reply):
        match = re.search(r'CALL:\s*(\w+)\((.*)\)\s*$', reply.strip(), re.DOTALL)
        if not match:
            return None
        name, raw_args = match.group(1), match.group(2).strip()
        if not raw_args:
            return name, {}
        try:
            return name, json.loads(raw_args)
        except json.JSONDecodeError:
            return name, {}

    @staticmethod
    def _bind_context(name, kwargs, ctx: ReasoningContext):
        """Fill in the pieces of context the model shouldn't have to invent:
        the live `state` object for perception tools, and company_id/cycle_id
        for action tools."""
        kwargs = dict(kwargs)
        if name in ('get_clv_summary', 'get_churn_risks', 'get_forecast_trend',
                    'segment_customers', 'compare_period_over_period'):
            kwargs['state'] = ctx.state
        if name in ('create_alert', 'log_recommendation'):
            kwargs['company_id'] = ctx.company_id
        if name == 'log_recommendation':
            kwargs['cycle_id'] = ctx.cycle_id
        return kwargs

    @staticmethod
    def _summarize_state(state):
        parts = []
        for key in ('clv_results', 'result_df', 'customer_df', 'forecast_data'):
            df = state.get(key)
            parts.append(f"{key}: {'present, ' + str(len(df)) + ' rows' if df is not None else 'not available'}")
        churn = state.get('churn_results')
        churn_df = churn.get('churn_predictions') if isinstance(churn, dict) else None
        parts.append(f"churn_results: {'present, ' + str(len(churn_df)) + ' rows' if churn_df is not None else 'not available'}")
        return '\n'.join(parts)

    @staticmethod
    def _truncate(result, limit=200):
        text = json.dumps(result, default=str)
        return text if len(text) <= limit else text[:limit] + '...'
