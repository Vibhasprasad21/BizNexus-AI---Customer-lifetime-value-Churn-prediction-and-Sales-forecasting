"""Builds the normalized snapshot the agent's tools operate on, and persists
fresh analysis results so that snapshot is still available when nobody has
an active Streamlit session open (scheduled/background cycles).

Today, CLV/churn/sales-forecast results only live in st.session_state -
they are computed per browser session and never written to the `analyses`
table. That's fine for "Run Agent Now" from inside the app, but it means a
background thread with no session_state has nothing to perceive. The
persist_* helpers here are called right after each analysis page computes
its results, so the `analyses` table always has the latest snapshot per
company and the agent can perceive state either way.
"""
import json

import pandas as pd

from src.database import store


def _df_to_records(df):
    if df is None or len(df) == 0:
        return []
    return json.loads(df.to_json(orient='records', date_format='iso'))


def _records_to_df(records):
    return pd.DataFrame(records) if records else pd.DataFrame()


# --- Persist (called from the analysis pages right after computing results) ---

def persist_clv_results(company_id, dataset_id, clv_df):
    payload = {'records': _df_to_records(clv_df)}
    store.save_analysis_results(company_id, dataset_id, 'clv', payload)


def persist_churn_results(company_id, dataset_id, churn_predictions_df):
    payload = {'records': _df_to_records(churn_predictions_df)}
    store.save_analysis_results(company_id, dataset_id, 'churn', payload)


def persist_forecast_results(company_id, dataset_id, forecast_df, growth_rate=None, volatility=None):
    payload = {
        'records': _df_to_records(forecast_df),
        'growth_rate_pct': growth_rate,
        'volatility': volatility,
    }
    store.save_analysis_results(company_id, dataset_id, 'sales_forecast', payload)


# --- Build (called by the agent before every reasoning cycle) -------------

def build_state(company_id, session_state=None):
    """Return a plain dict with the data the agent's tools expect:
    clv_results (DataFrame), churn_results ({'churn_predictions': DataFrame}),
    result_df, customer_df, forecast_data (DataFrame), plus whatever
    precomputed risk flags the analysis pages already left in session_state.

    Prefers a live Streamlit session's in-memory results (freshest, and free
    of a DB round-trip); falls back to the latest persisted analysis per
    company for any piece that's missing, so the agent still has something
    to perceive when run outside an active session.
    """
    ss = session_state if session_state is not None else {}

    def _get(key):
        try:
            return ss[key] if key in ss else None
        except Exception:
            return None

    source = 'live_session'

    clv_results = _get('clv_results')
    if clv_results is None:
        analysis = store.get_latest_analysis(company_id, 'clv')
        if analysis and analysis.get('results'):
            clv_results = _records_to_df(analysis['results'].get('records'))
            source = 'persisted'

    churn_results = _get('churn_results')
    if not (isinstance(churn_results, dict) and 'churn_predictions' in churn_results):
        analysis = store.get_latest_analysis(company_id, 'churn')
        if analysis and analysis.get('results'):
            churn_results = {'churn_predictions': _records_to_df(analysis['results'].get('records'))}
            source = 'persisted'
        else:
            churn_results = None

    forecast_data = _get('forecast_data')
    growth_rate = _get('sales_growth_rate')
    volatility = _get('sales_volatility')
    if forecast_data is None:
        analysis = store.get_latest_analysis(company_id, 'sales_forecast')
        if analysis and analysis.get('results'):
            forecast_data = _records_to_df(analysis['results'].get('records'))
            if 'Date' in forecast_data.columns:
                forecast_data['Date'] = pd.to_datetime(forecast_data['Date'], errors='coerce')
            growth_rate = growth_rate if growth_rate is not None else analysis['results'].get('growth_rate_pct')
            volatility = volatility if volatility is not None else analysis['results'].get('volatility')
            source = 'persisted'

    result_df = _get('result_df')
    customer_df = _get('customer_df')
    if result_df is None or customer_df is None:
        datasets = store.get_datasets(company_id)
        if datasets.get('success') and datasets['datasets']:
            latest_id = datasets['datasets'][0]['id']
            ds = store.get_dataset(company_id, latest_id)
            if ds.get('success'):
                df = ds['dataset']['dataframe']
                if result_df is None:
                    result_df = df
                if customer_df is None:
                    customer_df = df
                source = 'persisted'

    return {
        'company_id': company_id,
        'source': source,
        'clv_results': clv_results,
        'churn_results': churn_results,
        'result_df': result_df,
        'customer_df': customer_df,
        'forecast_data': forecast_data,
        'sales_growth_rate': growth_rate,
        'sales_volatility': volatility,
    }
