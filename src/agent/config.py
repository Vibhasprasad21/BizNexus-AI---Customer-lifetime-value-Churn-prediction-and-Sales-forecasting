"""Agent configuration: signal thresholds and the reasoning backend switch.

REASONING_BACKEND is the one flag that decides how much of the agent is
rule-based vs. LLM-backed - see src/agent/reasoning/ for the two
implementations. It defaults to 'rule_based' (100% local, no external calls,
deployable anywhere) and can be switched to 'ollama' to hand REASON/PLAN over
to a local LLM, provided Ollama is installed and reachable wherever the app
runs. It is never 'ollama' by default; nothing in this codebase calls out to
a network LLM API.
"""
import os


REASONING_BACKEND = os.environ.get('BIZNEXUS_AGENT_BACKEND', 'rule_based')  # 'rule_based' | 'ollama'

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('BIZNEXUS_OLLAMA_MODEL', 'llama3')

# Signal thresholds for the rule-based reasoning engine. Each maps directly
# to a tool call in src/agent/reasoning/rule_based.py.
THRESHOLDS = {
    # churn_spike: a customer counts as "newly at risk" once churn probability
    # crosses this line, and the signal only fires once `min_new_at_risk`
    # such customers appeared since the last cycle.
    'churn_risk_threshold': 0.6,
    'churn_min_new_at_risk': 3,

    # clv_drop: total portfolio CLV falling by more than this fraction
    # since the last recorded snapshot fires the signal.
    'clv_drop_pct': 0.05,

    # revenue_dip: forecast growth rate below this (percent, can be negative)
    # fires the signal; also fires on an actual month-over-month sales drop
    # of at least this magnitude.
    'revenue_dip_growth_pct': -10.0,
    'revenue_dip_mom_pct': -15.0,

    # seasonal_anomaly: forecast coefficient of variation above this fires
    # the signal (a proxy for "this doesn't look like the usual pattern").
    'seasonal_volatility': 0.3,
}

# How many rows of "who exactly" to attach to a recommendation/alert.
MAX_CUSTOMERS_IN_DETAIL = 10

# churn_spike self-throttles (it only acts on customers *new* since the last
# cycle). These three signals have no such natural diff - without a cooldown
# they'd re-fire, and re-alert, on every single cycle the condition persists.
THROTTLE_HOURS = {
    'clv_drop': 24,
    'revenue_dip': 24,
    'seasonal_anomaly': 24,
}
