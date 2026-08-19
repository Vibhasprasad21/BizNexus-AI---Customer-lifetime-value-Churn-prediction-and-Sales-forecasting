"""The Agent: owns the full perceive -> reason -> plan -> act -> observe loop.

This is the one entry point everything else uses - the "Run Agent Now"
button, the background scheduler, and (indirectly, for perception only) the
natural-language front-end all go through Agent.run_cycle().

  PERCEIVE  build_state() reads whatever CLV/churn/forecast/transaction data
            is currently available (live session or last persisted analysis).
  REASON +  delegated to a ReasoningBackend (rule_based by default - see
  PLAN +    src/agent/config.py to switch). The backend decides what's
  ACT       significant and chains tool calls to act on it. This is the one
            piece that's a genuine swap point between "coded rules" and
            "LLM judgment" - everything else in this file is the same
            either way.
  OBSERVE   re-checks past open recommendations against the state just
            perceived, so later cycles (and the Activity Log) can show
            whether a past flag was actually right.

Every step writes to agent_events (src/agent/memory.py) as it happens, so
the trail shown on the AI Agent page is a live record, not a reconstruction.
"""
from src.agent import config, data_access, memory, tools
from src.agent.reasoning import get_backend
from src.agent.reasoning.base import ReasoningContext


class Agent:
    def __init__(self, reasoning_backend=None):
        self.backend_name = reasoning_backend or config.REASONING_BACKEND
        self.backend = get_backend(self.backend_name)

    def run_cycle(self, company_id, session_state=None, trigger='manual'):
        """Run one full cycle for a company and return a summary dict
        (cycle_id, counts, the reasoning trail, and where the perceived data
        came from). `session_state` is st.session_state when called from an
        interactive page, or None for a scheduled/background run."""
        cycle_id = memory.start_cycle(company_id, trigger=trigger, reasoning_backend=self.backend_name)

        def log(step, message, signal_type=None, data=None):
            memory.log_event(cycle_id, company_id, step, message, signal_type=signal_type, data=data)

        # PERCEIVE
        state = data_access.build_state(company_id, session_state=session_state)
        log('perceive', self._describe_state(state))

        ctx = ReasoningContext(state=state, company_id=company_id, cycle_id=cycle_id, log=log)

        # REASON + PLAN + ACT
        self.backend.run_cycle(ctx)

        # OBSERVE
        observed = self._observe(company_id, state, log)

        summary = (f"{ctx.signals_found} signal(s) found, {ctx.actions_taken} action(s) taken, "
                   f"{observed} past recommendation(s) re-checked.")
        memory.finish_cycle(cycle_id, ctx.signals_found, ctx.actions_taken, summary)

        return {
            'cycle_id': cycle_id,
            'signals_found': ctx.signals_found,
            'actions_taken': ctx.actions_taken,
            'observed': observed,
            'summary': summary,
            'trail': memory.get_events_for_cycle(cycle_id),
            'state_source': state.get('source'),
            'backend': self.backend_name,
        }

    @staticmethod
    def _describe_state(state):
        bits = []
        clv = state.get('clv_results')
        bits.append(f"CLV data for {len(clv)} customers" if clv is not None and len(clv) else "no CLV data")

        churn = state.get('churn_results')
        churn_df = churn.get('churn_predictions') if isinstance(churn, dict) else None
        bits.append(f"churn predictions for {len(churn_df)} customers"
                    if churn_df is not None and len(churn_df) else "no churn predictions")

        forecast = state.get('forecast_data')
        bits.append(f"a {len(forecast)}-period sales forecast"
                    if forecast is not None and len(forecast) else "no sales forecast")

        return f"Perceived state (source: {state.get('source')}): " + ", ".join(bits) + "."

    def _observe(self, company_id, state, log):
        """Re-check past open recommendations against current data: was the
        flagged risk still there, or did it resolve? Only churn-spike
        recommendations carry enough structure (a target customer list plus
        a re-checkable threshold) to auto-observe today; CLV/revenue signals
        are compared cycle-to-cycle instead, via the snapshot table."""
        pending = memory.get_recommendations_needing_observation(
            company_id, signal_type='churn_spike', min_age_hours=1
        )
        if not pending:
            return 0

        merged = tools._merged_customer_view(state)  # internal reuse within the package
        churn_col = tools._churn_col(merged) if merged is not None else None
        if merged is None or not churn_col:
            return 0

        observed = 0
        for rec in pending:
            target_ids = {c['customer_id'] for c in rec['target_customers']}
            if not target_ids:
                continue

            still_at_risk = merged[
                merged['Customer ID'].isin(target_ids)
                & (merged[churn_col] > config.THRESHOLDS['churn_risk_threshold'])
            ]

            if len(still_at_risk) == 0:
                outcome = 'resolved'
                note = "None of the flagged customers are still above the churn threshold."
            elif len(still_at_risk) == len(target_ids):
                outcome = 'confirmed'
                note = "All flagged customers are still above the churn threshold."
            else:
                outcome = 'partially_confirmed'
                note = f"{len(still_at_risk)}/{len(target_ids)} are still above the churn threshold."

            memory.record_outcome(rec['id'], outcome, note)
            log('observe', f"Re-checked \"{rec['title']}\": {outcome.replace('_', ' ')} - {note}",
                signal_type=rec['signal_type'], data={'recommendation_id': rec['id'], 'outcome': outcome})
            observed += 1

        return observed
