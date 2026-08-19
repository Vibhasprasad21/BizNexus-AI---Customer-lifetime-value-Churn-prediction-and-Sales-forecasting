"""Natural-language front-end to the agent's tools.

This replaces src/chatbot/assistant.py. The old assistant matched a query
against keyword patterns and then ran its OWN hand-written logic per intent -
duplicate business logic that could (and did) drift from what the autonomous
agent actually does. This router still matches intents with keywords/regex
(explicitly rule-based - see the capability boundary note below), but every
handler answers by calling the SAME tool functions in src/agent/tools.py
that the autonomous reasoning engine uses. One source of truth for "what
counts as high churn risk" or "what's a retention plan for a Premium
customer", whether a human asked in chat or the agent decided on its own.

Capability boundary: intent routing here is regex/keyword matching, same
mechanism as the reasoning engine's thresholds. It handles the phrasings
below well and will misroute or fall back to a generic answer for novel or
ambiguous phrasing a keyword list wasn't written for - a genuinely
open-ended question ("why do you think churn is up?") needs judgment a
fixed pattern list can't provide. Swapping REASONING_BACKEND to 'ollama'
(src/agent/config.py) would let a local LLM route and answer instead of
this file, without changing any tool.
"""
import re

import plotly.express as px

from src.agent import config, memory, tools


def _merged_or_none(state):
    return tools._merged_customer_view(state)


def route(query, state):
    """Given a user's natural-language query and the current perceived
    state, decide which tool(s) to call and return a chat response dict:
    {'text': str, 'visualization': plotly figure or None}."""
    q = query.lower().strip()

    if _is_agent_status_query(q):
        return _handle_agent_status(state)
    if _is_customer_specific_query(q):
        return _handle_customer_specific(query, q, state)
    # Retention/strategy phrasing ("retention strategy for premium customers")
    # is checked ahead of the top-customers and churn-risk intents because it
    # can otherwise share trigger words with both ("premium customers",
    # "reduce churn") and lose to a more generic match.
    if _is_retention_query(q):
        return _handle_retention(query, q, state)
    if _is_top_customers_query(q):
        return _handle_top_customers(state)
    if _is_churn_risk_query(q):
        return _handle_churn_risk(state)
    if _is_forecast_query(q):
        return _handle_forecast(state)
    if _is_segment_query(q):
        return _handle_segments(q, state)

    return _handle_fallback()


# --- Intent detection (kept simple and explicit - see module docstring) -----

def _is_agent_status_query(q):
    return any(t in q for t in ['what have you found', 'agent status', 'recent alerts',
                                 'what did you notice', 'recommendations', 'what changed'])


def _is_customer_specific_query(q):
    patterns = [r'clv of', r'details of', r'customer profile', r'customer \d+',
                r'customer id', r'profile of', r'churn probability for',
                r'lifetime value of', r'value of customer', r'info about',
                r'information on', r'churn risk for']
    return any(re.search(p, q) for p in patterns)


def _is_top_customers_query(q):
    if re.search(r'\btop\s*\d*\s*customers?\b', q):
        return True
    return any(t in q for t in ['most valuable', 'high value customer', 'best customer', 'highest clv'])


def _is_churn_risk_query(q):
    terms = ['churn risk', 'at risk', 'likely to churn', 'churn probability',
             'who will leave', 'customer churn', 'highest risk', 'churn rate']
    return any(t in q for t in terms) and not _is_customer_specific_query(q)


def _is_forecast_query(q):
    return any(t in q for t in ['sales trend', 'sales forecast', 'revenue projection',
                                 'future sales', 'projected revenue', 'forecast', 'prediction'])


def _is_segment_query(q):
    return any(t in q for t in ['compare', 'comparison', 'versus', ' vs ', 'diff', 'difference between'])


def _is_retention_query(q):
    return any(t in q for t in ['strategy', 'strategies', 'retain', 'retention',
                                 'prevent churn', 'reduce churn', 'recommend'])


# --- Handlers: each calls tools.py, none re-derives business logic ---------

def _handle_agent_status(state):
    company_id = state.get('company_id')
    recs = memory.get_recommendations(company_id, status='pending', limit=5)
    last_cycle = memory.get_last_cycle(company_id)

    lines = ["## Agent Status\n"]
    if last_cycle:
        lines.append(f"Last run: {last_cycle['started_at'][:16].replace('T', ' ')} "
                      f"({last_cycle.get('trigger')}) - {last_cycle.get('summary', '')}\n")
    else:
        lines.append("The agent hasn't run yet for this company. Use \"Run Agent Now\" on the AI Agent page.\n")

    if recs:
        lines.append("### Pending recommendations\n")
        for r in recs:
            lines.append(f"- **[{r['priority'].upper()}] {r['title']}**")
    else:
        lines.append("No pending recommendations right now.")

    return {'text': '\n'.join(lines), 'visualization': None}


def _handle_customer_specific(raw_query, q, state):
    identifier = _extract_identifier(raw_query)
    if not identifier:
        return {'text': "Could not identify a specific customer in your question. "
                         "Try including a customer ID or name.", 'visualization': None}

    profile = tools.find_customer(state, identifier)
    if not profile.get('found'):
        return {'text': profile.get('reason', 'Customer not found.'), 'visualization': None}

    lines = [f"## Customer Profile: {profile['name']}\n",
             f"**Customer ID:** {profile['customer_id']}",
             f"**Customer Lifetime Value:** ${profile['clv']:,.2f}"]
    if profile.get('value_tier'):
        lines.append(f"**Value Tier:** {profile['value_tier']}")
    if profile.get('churn_probability') is not None:
        lines.append(f"**Churn Probability:** {profile['churn_probability']:.1%}")
    if profile.get('clv_percentile') is not None:
        lines.append(f"**CLV Percentile:** {profile['clv_percentile']:.0f}%")

    if any(t in q for t in ['recommend', 'suggestion', 'retention', 'strategy']) or \
       (profile.get('churn_probability') or 0) > 0.5:
        issue_type = 'churn_high_value' if profile.get('value_tier') in ('Premium', 'High') else 'churn_general'
        plan = tools.generate_action_plan(issue_type, context={'customer': profile['name']})
        lines.append(f"\n### {plan['headline']}\n")
        lines.extend(f"{i}. {step}" for i, step in enumerate(plan['steps'], 1))

    return {'text': '\n'.join(lines), 'visualization': None}


def _handle_top_customers(state):
    top = tools.get_top_customers(state, n=10)
    if not top.get('available'):
        return {'text': top.get('reason', 'No CLV data available.'), 'visualization': None}

    summary = tools.get_clv_summary(state)
    lines = ["## Top 10 Most Valuable Customers\n"]
    for c in top['customers']:
        risk = f"{c['churn_probability']:.1%}" if c['churn_probability'] is not None else 'N/A'
        lines.append(f"- **{c['name']}** (ID {c['customer_id']}) - ${c['clv']:,.2f} CLV, "
                      f"{c.get('value_tier', 'N/A')} tier, {risk} churn risk")

    if summary.get('available'):
        lines.append(f"\n### Portfolio context\n"
                      f"- Average CLV: ${summary['avg_clv']:,.2f}\n"
                      f"- Total customer value: ${summary['total_clv']:,.2f}\n"
                      f"- Customers analyzed: {summary['customer_count']}")

    fig = px.bar(
        {'name': [c['name'] for c in top['customers']], 'clv': [c['clv'] for c in top['customers']]},
        x='name', y='clv', title='Top 10 Customers by CLV', labels={'clv': 'CLV ($)', 'name': 'Customer'},
    )
    return {'text': '\n'.join(lines), 'visualization': fig}


def _handle_churn_risk(state):
    risks = tools.get_churn_risks(state, threshold=0.5)
    if not risks.get('available'):
        return {'text': risks.get('reason', 'No churn data available.'), 'visualization': None}

    lines = [f"## {risks['count']} Customers at High Churn Risk (>50%)\n",
             f"**Total value at risk:** ${risks['total_value_at_risk']:,.2f}\n"]
    for c in risks['customers'][:10]:
        lines.append(f"- **{c['name']}** - {c['churn_probability']:.1%} risk, ${c['clv']:,.2f} CLV, "
                      f"{c.get('value_tier', 'N/A')} tier")

    segments = tools.segment_customers(state, criteria='value_tier')
    fig = None
    if segments.get('available'):
        segs = segments['segments']
        fig = px.bar(
            {'segment': [s['segment'] for s in segs],
             'avg_churn': [s.get('avg_churn_probability', 0) for s in segs]},
            x='segment', y='avg_churn', title='Average Churn Probability by Value Tier',
            labels={'avg_churn': 'Avg. Churn Probability', 'segment': 'Value Tier'},
        )

    return {'text': '\n'.join(lines), 'visualization': fig}


def _handle_forecast(state):
    trend = tools.get_forecast_trend(state)
    if not trend.get('available'):
        return {'text': trend.get('reason', 'No sales forecast available.'), 'visualization': None}

    lines = ["## Sales Forecast\n"]
    if trend.get('growth_rate_pct') is not None:
        lines.append(f"**Growth trend:** {trend['growth_rate_pct']:+.1f}% over the forecast horizon")
    if trend.get('avg_forecast') is not None:
        lines.append(f"**Average forecasted period:** ${trend['avg_forecast']:,.2f}")
    if trend.get('volatility') is not None:
        lines.append(f"**Volatility (coefficient of variation):** {trend['volatility']:.2f}")

    mom = tools.compare_period_over_period(state, metric='sales')
    if mom.get('available'):
        lines.append(f"\n**{mom['current_period']} vs {mom['previous_period']}:** "
                      f"${mom['current_value']:,.2f} vs ${mom['previous_value']:,.2f} "
                      f"({mom['change_pct']:+.1f}%)")

    fig = None
    forecast_df = state.get('forecast_data')
    if forecast_df is not None and 'Date' in forecast_df.columns and 'Forecast' in forecast_df.columns:
        fig = px.line(forecast_df, x='Date', y='Forecast', title='Sales Forecast')

    return {'text': '\n'.join(lines), 'visualization': fig}


def _handle_segments(q, state):
    criteria = 'churn_band' if 'churn' in q or 'risk' in q else 'value_tier'
    segments = tools.segment_customers(state, criteria=criteria)
    if not segments.get('available'):
        return {'text': segments.get('reason', 'No segment data available.'), 'visualization': None}

    lines = [f"## Segment Comparison ({segments['criteria']})\n",
             "| Segment | Count | Avg CLV | Total CLV | Avg Churn Risk |",
             "|---|---|---|---|---|"]
    for s in segments['segments']:
        churn = f"{s['avg_churn_probability']:.1%}" if 'avg_churn_probability' in s else 'N/A'
        lines.append(f"| {s['segment']} | {s['count']} | ${s['avg_clv']:,.2f} | ${s['total_clv']:,.2f} | {churn} |")

    fig = px.bar(
        {'segment': [s['segment'] for s in segments['segments']],
         'avg_clv': [s['avg_clv'] for s in segments['segments']]},
        x='segment', y='avg_clv', title='Average CLV by Segment',
    )
    return {'text': '\n'.join(lines), 'visualization': fig}


def _handle_retention(raw_query, q, state):
    segment_match = re.search(r'for\s+(\w+)[\s-]+(?:segment|tier|customers)', raw_query, re.IGNORECASE)
    segment = segment_match.group(1).capitalize() if segment_match else None

    risks = tools.get_churn_risks(state, threshold=0.5)
    issue_type = 'churn_general'
    context = {}
    if risks.get('available'):
        high_value_at_risk = [c for c in risks['customers'] if c.get('value_tier') in ('Premium', 'High')]
        if segment in ('Premium', 'High') or (not segment and high_value_at_risk):
            issue_type = 'churn_high_value'
        context = {'segment': segment, 'total_at_risk': risks['count'],
                   'value_at_risk': risks['total_value_at_risk']}

    plan = tools.generate_action_plan(issue_type, context=context)
    lines = [f"## {plan['headline']}\n"]
    lines.extend(f"{i}. {step}" for i, step in enumerate(plan['steps'], 1))

    if risks.get('available'):
        lines.append(f"\n### Current context\n- {risks['count']} customers above 50% churn risk "
                      f"(${risks['total_value_at_risk']:,.2f} CLV at risk)")

    return {'text': '\n'.join(lines), 'visualization': None}


def _handle_fallback():
    return {
        'text': ("I can answer questions about specific customers, top customers by value, "
                  "churn risk, sales forecasts, segment comparisons, and retention strategies - "
                  "all backed by the same tools the autonomous agent uses. Try asking things like "
                  "\"who's at risk of churning\", \"top 10 customers\", \"sales forecast\", or "
                  "\"retention strategy for premium customers\"."),
        'visualization': None,
    }


def _extract_identifier(query):
    patterns = [
        r'(?:clv|details|profile|value|churn|information|info)\s+(?:of|for|about)\s+([a-zA-Z\s]+)',
        r'customer\s+([a-zA-Z\s]+)',
        r"([a-zA-Z\s]+)'s\s+(?:clv|profile|details|value|churn)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    id_match = re.search(r'customer\s*(?:#|id|number|)?\s*(\d+)', query, re.IGNORECASE)
    if id_match:
        return id_match.group(1)
    return None
