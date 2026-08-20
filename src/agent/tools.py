"""Agent tools.

Every function here is a plain, independently-callable Python function with
a docstring describing what it does and when to use it. That docstring is
the tool's "description" as seen by both the rule-based reasoning engine
(src/agent/reasoning/rule_based.py) and, if enabled, an LLM planner
(src/agent/reasoning/ollama_backend.py) - so keep it accurate.

Tools are split into two kinds:
  - perception tools (get_*, segment_customers, compare_period_over_period):
    read-only, operate on the `state` dict produced by
    src/agent/data_access.build_state().
  - action tools (create_alert, log_recommendation, generate_action_plan):
    have side effects (write to the DB) or produce a plan to act on.

None of this calls an LLM. Given a `state`, these are deterministic.
"""
import pandas as pd

from src.agent import memory
from src.database.store import save_notification


TOOL_DOCS = {}


def tool(func):
    """Register a function's docstring for introspection by the NL router
    and any future LLM-backed reasoning backend."""
    TOOL_DOCS[func.__name__] = (func.__doc__ or '').strip()
    return func


# --- Internal helpers -------------------------------------------------------

def _canonical_clv(clv_df):
    """Alias 'CLV' to the dedicated CLV Analysis page's discounted figure.

    clv_results carries both 'CLV' (a quick bootstrap estimate computed at
    upload time, before any horizon/discount-rate choice exists) and, once
    the CLV Analysis page has run, 'CLV_Adjusted' (that page's own
    prediction, properly discounted, honoring whatever time_horizon/
    discount_rate the user picked - the number the page itself displays as
    the headline figure). Every tool below reads a plain 'CLV' column, so
    this aliases it to CLV_Adjusted when available rather than silently
    using the cruder bootstrap number everywhere the agent reasons about
    value."""
    if clv_df is None or len(clv_df) == 0:
        return clv_df
    if 'CLV_Adjusted' in clv_df.columns:
        clv_df = clv_df.copy()
        clv_df['CLV'] = clv_df['CLV_Adjusted']
    return clv_df


def _merged_customer_view(state):
    """CLV + churn predictions joined on Customer ID - the common working table."""
    clv = _canonical_clv(state.get('clv_results'))
    churn = state.get('churn_results') or {}
    churn_df = churn.get('churn_predictions') if isinstance(churn, dict) else None

    if clv is None or len(clv) == 0:
        return None
    if churn_df is None or len(churn_df) == 0:
        return clv.copy()

    return pd.merge(clv, churn_df, on='Customer ID', how='left', suffixes=('', '_Churn'))


def _churn_col(df):
    """Pick the real churn model's column over the CLV page's crude fallback.

    clv_model.py always backfills a 'Churn_Probability' column onto CLV
    results if one isn't already there, using Churn_Prediction_90d or a
    binary Churn_Label as a stand-in (see enhanced_main_clv_analysis) - long
    before the dedicated Churn Prediction page's XGBoost model has run. After
    _merged_customer_view's pd.merge(..., suffixes=('', '_Churn')), that
    fallback survives as the unsuffixed 'Churn_Probability' column, while the
    real churn model's output becomes 'Churn_Probability_Churn'. Checking the
    suffixed name first means the agent reasons over actual model output
    whenever it's available, instead of a same-named stand-in - confirmed via
    QA that picking the wrong one silently degrades every churn number the
    agent and NL assistant surface into a degenerate 0%/100% recency flag.
    """
    if 'Churn_Probability_Churn' in df.columns:
        return 'Churn_Probability_Churn'
    if 'Churn_Probability' in df.columns:
        return 'Churn_Probability'
    return None


# --- Perception tools --------------------------------------------------------

@tool
def get_clv_summary(state):
    """Summarize the current CLV distribution: customer count, average/median/
    total value, and the breakdown by Value_Tier (Low/Medium/High/Premium).
    Use this any time you need a top-level read on how much the customer base
    is worth right now, before deciding whether a CLV signal is significant."""
    clv = _canonical_clv(state.get('clv_results'))
    if clv is None or len(clv) == 0:
        return {'available': False, 'reason': 'No CLV analysis available yet.'}

    summary = {
        'available': True,
        'customer_count': int(len(clv)),
        'avg_clv': float(clv['CLV'].mean()),
        'median_clv': float(clv['CLV'].median()),
        'total_clv': float(clv['CLV'].sum()),
    }
    if 'Value_Tier' in clv.columns:
        tiers = clv['Value_Tier'].value_counts().to_dict()
        summary['tier_counts'] = {str(k): int(v) for k, v in tiers.items()}
        tier_value = clv.groupby('Value_Tier', observed=True)['CLV'].sum().to_dict()
        summary['tier_total_value'] = {str(k): float(v) for k, v in tier_value.items()}
    return summary


@tool
def get_churn_risks(state, threshold=0.5):
    """List customers whose churn probability is above `threshold` (default
    0.5), sorted by CLV descending, with the count and total revenue at risk.
    Use this to find out who is likely to leave and how much that would cost -
    it's the starting point for any churn-related signal."""
    merged = _merged_customer_view(state)
    if merged is None:
        return {'available': False, 'reason': 'No churn/CLV analysis available yet.'}

    col = _churn_col(merged)
    if not col:
        return {'available': False, 'reason': 'Churn predictions not found in current data.'}

    at_risk = merged[merged[col] > threshold].copy()
    if 'CLV' in at_risk.columns:
        at_risk = at_risk.sort_values('CLV', ascending=False)

    customers = [{
        'customer_id': row.get('Customer ID'),
        'name': row.get('Customer Name', f"Customer {row.get('Customer ID')}"),
        'churn_probability': float(row.get(col, 0) or 0),
        'clv': float(row.get('CLV', 0) or 0),
        'value_tier': str(row.get('Value_Tier')) if row.get('Value_Tier') is not None else None,
    } for _, row in at_risk.iterrows()]

    return {
        'available': True,
        'threshold': threshold,
        'count': len(customers),
        'total_value_at_risk': float(at_risk['CLV'].sum()) if 'CLV' in at_risk.columns else 0.0,
        'customers': customers,
    }


@tool
def get_forecast_trend(state):
    """Read the sales forecast trend: growth rate over the forecast horizon,
    volatility, average/min/max forecasted period. Use this to check whether
    revenue is projected to rise, fall, or swing before flagging a revenue
    signal."""
    forecast = state.get('forecast_data')
    growth_rate = state.get('sales_growth_rate')
    volatility = state.get('sales_volatility')

    if forecast is None or len(forecast) == 0:
        if growth_rate is None:
            return {'available': False, 'reason': 'No sales forecast available yet.'}
        return {
            'available': True,
            'growth_rate_pct': float(growth_rate),
            'volatility': float(volatility) if volatility is not None else None,
        }

    df = forecast.copy()
    if growth_rate is None and 'Forecast' in df.columns and len(df) >= 2:
        first, last = df['Forecast'].iloc[0], df['Forecast'].iloc[-1]
        growth_rate = (last / first - 1) * 100 if first else 0.0

    if volatility is None and 'Forecast' in df.columns and df['Forecast'].mean():
        volatility = float(df['Forecast'].std() / df['Forecast'].mean())

    result = {
        'available': True,
        'growth_rate_pct': float(growth_rate) if growth_rate is not None else None,
        'volatility': float(volatility) if volatility is not None else None,
        'periods': int(len(df)),
    }
    if 'Forecast' in df.columns:
        result['avg_forecast'] = float(df['Forecast'].mean())
        result['min_forecast'] = float(df['Forecast'].min())
        result['max_forecast'] = float(df['Forecast'].max())
    return result


@tool
def segment_customers(state, criteria='value_tier'):
    """Group customers by `criteria` - 'value_tier' (Low/Medium/High/Premium)
    or 'churn_band' (Low/Medium/High Risk) - and report per-segment count,
    average CLV, total CLV, and average churn probability. Use this after
    finding a risk, to break it down by segment before drafting per-segment
    retention actions."""
    merged = _merged_customer_view(state)
    if merged is None:
        return {'available': False, 'reason': 'No CLV analysis available yet.'}

    churn_col = _churn_col(merged)

    if criteria == 'churn_band':
        if not churn_col:
            return {'available': False, 'reason': 'No churn data to segment by.'}
        merged = merged.copy()
        merged['_segment'] = pd.cut(
            merged[churn_col], bins=[-0.01, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        group_col = '_segment'
    else:
        if 'Value_Tier' not in merged.columns:
            return {'available': False, 'reason': 'No Value_Tier column to segment by.'}
        group_col = 'Value_Tier'

    if 'CLV' not in merged.columns:
        return {'available': False, 'reason': 'No CLV column to aggregate.'}

    agg = {'CLV': ['count', 'mean', 'sum']}
    if churn_col:
        agg[churn_col] = 'mean'

    grouped = merged.groupby(group_col, observed=True).agg(agg)
    grouped.columns = ['_'.join(c).strip('_') for c in grouped.columns]
    grouped = grouped.reset_index()

    segments = []
    for _, row in grouped.iterrows():
        seg = {
            'segment': str(row[group_col]),
            'count': int(row['CLV_count']),
            'avg_clv': float(row['CLV_mean']),
            'total_clv': float(row['CLV_sum']),
        }
        if churn_col:
            seg['avg_churn_probability'] = float(row.get(f'{churn_col}_mean', 0) or 0)
        segments.append(seg)

    return {'available': True, 'criteria': criteria, 'segments': segments}


@tool
def find_customer(state, identifier):
    """Look up a single customer by numeric Customer ID or by (partial,
    case-insensitive) name match against Customer Name. Returns their full
    merged CLV+churn profile, or `found: False` if no match. Use this for
    any question about one named or numbered customer."""
    merged = _merged_customer_view(state)
    if merged is None:
        return {'found': False, 'reason': 'No CLV analysis available yet.'}

    identifier = str(identifier).strip()
    match = None

    if identifier.isdigit():
        id_match = merged[merged['Customer ID'] == int(identifier)]
        if not id_match.empty:
            match = id_match.iloc[0]

    if match is None and 'Customer Name' in merged.columns:
        exact = merged[merged['Customer Name'].str.lower() == identifier.lower()]
        if not exact.empty:
            match = exact.iloc[0]
        else:
            contains = merged[merged['Customer Name'].str.lower().str.contains(identifier.lower(), na=False)]
            if not contains.empty:
                match = contains.iloc[0]

    if match is None:
        return {'found': False, 'reason': f"No customer matching '{identifier}'."}

    churn_col = _churn_col(merged)
    profile = {
        'found': True,
        'customer_id': match.get('Customer ID'),
        'name': match.get('Customer Name', f"Customer {match.get('Customer ID')}"),
        'clv': float(match.get('CLV', 0) or 0),
        'value_tier': str(match.get('Value_Tier')) if match.get('Value_Tier') is not None else None,
        'churn_probability': float(match.get(churn_col, 0) or 0) if churn_col else None,
        'clv_percentile': float((merged['CLV'] < match.get('CLV', 0)).mean() * 100) if 'CLV' in merged.columns else None,
    }
    return profile


@tool
def get_top_customers(state, n=10, by='CLV'):
    """Return the top `n` customers ranked by `by` (default 'CLV'). Use this
    for "most valuable customers" style questions, or to attach a concrete
    customer list to a high-value churn/CLV signal."""
    merged = _merged_customer_view(state)
    if merged is None or by not in merged.columns:
        return {'available': False, 'reason': 'No CLV analysis available yet.'}

    churn_col = _churn_col(merged)
    top = merged.nlargest(n, by)
    customers = [{
        'customer_id': row.get('Customer ID'),
        'name': row.get('Customer Name', f"Customer {row.get('Customer ID')}"),
        'clv': float(row.get('CLV', 0) or 0),
        'value_tier': str(row.get('Value_Tier')) if row.get('Value_Tier') is not None else None,
        'churn_probability': float(row.get(churn_col, 0) or 0) if churn_col else None,
    } for _, row in top.iterrows()]

    return {'available': True, 'by': by, 'customers': customers}


@tool
def compare_period_over_period(state, metric='sales'):
    """Compare the two most recent periods for `metric` - 'sales' (monthly
    transaction totals) or 'clv' (current CLV snapshot vs. the last one the
    agent recorded in its own memory). Use this to quantify a dip or spike
    with a number before deciding how urgent it is."""
    if metric == 'clv':
        clv = _canonical_clv(state.get('clv_results'))
        if clv is None or len(clv) == 0 or 'CLV' not in clv.columns:
            return {'available': False, 'reason': 'No CLV data available.'}
        current_total = float(clv['CLV'].sum())
        current_avg = float(clv['CLV'].mean())
        previous = memory.get_last_snapshot(state.get('company_id'), 'total_clv')
        if not previous:
            return {
                'available': True, 'has_baseline': False,
                'current_total_clv': current_total, 'current_avg_clv': current_avg,
            }
        change_pct = (current_total / previous['value'] - 1) * 100 if previous['value'] else 0.0
        return {
            'available': True, 'has_baseline': True,
            'current_total_clv': current_total, 'current_avg_clv': current_avg,
            'previous_total_clv': previous['value'], 'previous_recorded_at': previous['recorded_at'],
            'change_pct': change_pct,
        }

    result_df = state.get('result_df')
    if result_df is None or len(result_df) == 0:
        return {'available': False, 'reason': 'No transaction data available.'}

    date_col = next((c for c in result_df.columns if any(t in c.lower() for t in ['date', 'time', 'day'])), None)
    amount_col = next((c for c in result_df.columns if any(t in c.lower() for t in ['amount', 'price', 'revenue', 'sales'])), None)
    if not date_col or not amount_col:
        return {'available': False, 'reason': 'Could not find date/amount columns in transaction data.'}

    df = result_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    if df.empty:
        return {'available': False, 'reason': 'No valid transaction dates.'}

    monthly = df.groupby(pd.Grouper(key=date_col, freq='ME'))[amount_col].sum().reset_index()
    monthly = monthly.sort_values(date_col, ascending=False)

    if len(monthly) < 2:
        return {'available': False, 'reason': 'Not enough historical periods to compare.'}

    current, previous = monthly.iloc[0], monthly.iloc[1]
    change_pct = (current[amount_col] / previous[amount_col] - 1) * 100 if previous[amount_col] else 0.0

    return {
        'available': True,
        'current_period': current[date_col].strftime('%B %Y'),
        'previous_period': previous[date_col].strftime('%B %Y'),
        'current_value': float(current[amount_col]),
        'previous_value': float(previous[amount_col]),
        'change_pct': float(change_pct),
    }


# --- Action tools ------------------------------------------------------------

@tool
def create_alert(company_id, message, severity='medium', alert_type='agent'):
    """Create an in-app alert visible in the app's notification panel.
    `severity` is one of 'low' | 'medium' | 'high' and maps directly to
    notification priority. Use this once a signal has been reasoned about and
    the business owner should be told about it now."""
    result = save_notification(company_id, notification_type=alert_type, message=message, priority=severity)
    return {'success': bool(result.get('success')), 'notification_id': result.get('notification_id')}


@tool
def log_recommendation(company_id, text, target_customers=None, title=None,
                        priority='medium', signal_type=None, cycle_id=None):
    """Write a recommendation to the database as a pending action item the
    business owner can later approve or dismiss on the AI Agent page. Use
    this as the durable record of what the agent proposes to do, once a plan
    has been decided - this is what makes a recommendation persist beyond a
    single reasoning cycle."""
    rec_id = memory.log_recommendation(
        company_id,
        title=title or (text[:80] + ('...' if len(text) > 80 else '')),
        body=text, cycle_id=cycle_id, signal_type=signal_type,
        target_customers=target_customers, priority=priority,
    )
    return {'success': True, 'recommendation_id': rec_id}


_ACTION_PLAN_TEMPLATES = {
    'churn_high_value': {
        'headline': 'VIP retention play for high-value customers at risk',
        'steps': [
            'Executive outreach: schedule a business review with senior leadership',
            "Build a custom success plan around this customer's usage patterns",
            'Offer a loyalty package with premium incentives',
            'Assign a named account representative for direct support',
            'Audit underutilized features that could demonstrate more value',
        ],
    },
    'churn_general': {
        'headline': 'Re-engagement campaign for at-risk customers',
        'steps': [
            'Enroll in an automated re-engagement email sequence',
            'Send a short satisfaction survey to surface pain points',
            'Share targeted usage guidance for underused features',
            'Offer a limited-time renewal incentive',
            'Share a relevant success story or case study',
        ],
    },
    'clv_decline': {
        'headline': 'Reverse a declining customer lifetime value trend',
        'steps': [
            'Identify which segment is dragging the average down',
            'Review recent pricing or packaging changes for that segment',
            'Launch a targeted upsell/cross-sell campaign for mid-tier customers',
            'Check for a drop in purchase frequency vs. the prior period',
            'Re-run CLV after the next data refresh to confirm direction',
        ],
    },
    'revenue_dip': {
        'headline': 'Respond to a projected revenue decline',
        'steps': [
            'Confirm whether the dip is broad-based or concentrated in a segment',
            'Check customer count and average order value for the affected period',
            'Consider a short-term promotion to offset the projected shortfall',
            'Flag the forecast to sales/finance for planning',
            'Re-check the forecast next cycle to see if the trend holds',
        ],
    },
    'seasonal_anomaly': {
        'headline': 'Investigate an unexpected deviation from the seasonal pattern',
        'steps': [
            'Compare the current period against the same period last year, if available',
            'Check for one-off causes: promotions, outages, data gaps',
            'Decide whether this changes near-term forecast assumptions',
            'Monitor the next period before escalating further',
        ],
    },
}


@tool
def generate_action_plan(issue_type, context=None):
    """Produce a structured retention/growth action plan for `issue_type`
    (one of: churn_high_value, churn_general, clv_decline, revenue_dip,
    seasonal_anomaly). `context` is an optional dict of numbers (counts,
    dollar amounts) attached to the plan for reference. This is a fixed
    rule-based template lookup, not a generated write-up - it will always
    return the same steps for the same issue_type regardless of the specific
    customers or numbers involved."""
    template = _ACTION_PLAN_TEMPLATES.get(issue_type, _ACTION_PLAN_TEMPLATES['churn_general'])
    return {
        'issue_type': issue_type,
        'headline': template['headline'],
        'steps': list(template['steps']),
        'context': context or {},
    }


TOOL_REGISTRY = {
    'get_clv_summary': get_clv_summary,
    'get_churn_risks': get_churn_risks,
    'get_forecast_trend': get_forecast_trend,
    'segment_customers': segment_customers,
    'compare_period_over_period': compare_period_over_period,
    'find_customer': find_customer,
    'get_top_customers': get_top_customers,
    'create_alert': create_alert,
    'log_recommendation': log_recommendation,
    'generate_action_plan': generate_action_plan,
}
