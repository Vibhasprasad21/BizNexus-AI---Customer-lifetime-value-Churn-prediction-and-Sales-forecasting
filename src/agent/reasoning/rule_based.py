"""The default reasoning backend: a scored, prioritized list of signals,
each evaluated with a threshold and, if it fires, resolved by chaining
several tool calls rather than returning a canned response.

This is deliberately NOT a lookup table. What happens for a given signal
depends on what earlier tool calls discover - e.g. the churn-spike signal
only files a "high-value VIP retention" recommendation for the tiers that
are actually represented among the newly-at-risk customers this cycle,
and only after diffing against last cycle's flagged set to find out what's
*new*. Two runs against different data take different paths through the
tool calls, even though the code driving them is fixed.

Capability boundary: every decision here is an explicit if/threshold check
against numbers the tools return. It handles the four signal types below
well, and nothing else - a genuinely novel question ("did the trade show
in March explain the July dip?") is outside what this backend can reason
about. That's the rule-based/LLM boundary the project README calls out;
see ollama_backend.py for the swap-in that would close this gap.
"""
from src.agent import config, memory, tools
from src.agent.reasoning.base import ReasoningBackend, ReasoningContext


class RuleBasedReasoningBackend(ReasoningBackend):
    name = 'rule_based'

    def run_cycle(self, ctx: ReasoningContext) -> None:
        # Priority order: protect at-risk revenue first, then portfolio value,
        # then top-line revenue trend, then softer anomaly signals.
        self._check_churn_spike(ctx)
        self._check_clv_drop(ctx)
        self._check_revenue_dip(ctx)
        self._check_seasonal_anomaly(ctx)

    # --- Signal 1: high-CLV customers newly crossing into high churn risk ---

    def _check_churn_spike(self, ctx: ReasoningContext):
        th = config.THRESHOLDS['churn_risk_threshold']
        risks = tools.get_churn_risks(ctx.state, threshold=th)

        if not risks.get('available'):
            ctx.log('reason', f"Churn spike check: {risks.get('reason', 'no data available')}.",
                    signal_type='churn_spike')
            return

        prev_recs = memory.get_last_cycle_recommendations_by_signal(ctx.company_id, 'churn_spike')
        previous_ids = {c['customer_id'] for rec in prev_recs for c in rec['target_customers']}
        current = risks['customers']
        newly_at_risk = [c for c in current if c['customer_id'] not in previous_ids]

        ctx.log(
            'reason',
            f"Checked churn risk at >{th:.0%}: {risks['count']} customers currently at risk "
            f"(${risks['total_value_at_risk']:,.0f} CLV). "
            f"{len(newly_at_risk)} of those are new since the last cycle.",
            signal_type='churn_spike',
            data={'threshold': th, 'total_at_risk': risks['count'], 'newly_at_risk': len(newly_at_risk)},
        )

        min_new = config.THRESHOLDS['churn_min_new_at_risk']
        if len(newly_at_risk) < min_new:
            return

        ctx.note_signal()
        by_tier = {}
        for c in newly_at_risk:
            by_tier.setdefault(c.get('value_tier') or 'Unknown', []).append(c)

        ctx.log(
            'plan',
            f"Plan: {len(newly_at_risk)} newly at-risk customers span {len(by_tier)} value tier(s) "
            f"({', '.join(f'{t}: {len(cs)}' for t, cs in by_tier.items())}). "
            f"Drafting a tailored retention plan per tier, logging recommendations, and raising an alert.",
            signal_type='churn_spike',
        )

        total_value = sum(c['clv'] for c in newly_at_risk)
        for tier, customers in by_tier.items():
            issue_type = 'churn_high_value' if tier in ('Premium', 'High') else 'churn_general'
            tier_value = sum(c['clv'] for c in customers)
            plan = tools.generate_action_plan(issue_type, context={
                'tier': tier, 'customer_count': len(customers), 'total_clv': tier_value,
            })
            body = self._format_plan(plan, customers)
            tools.log_recommendation(
                ctx.company_id, body,
                target_customers=customers,
                title=f"{plan['headline']} - {tier} tier ({len(customers)} customers)",
                priority='high' if issue_type == 'churn_high_value' else 'medium',
                signal_type='churn_spike', cycle_id=ctx.cycle_id,
            )
            ctx.log('act', f"Logged recommendation for {len(customers)} {tier}-tier at-risk customers "
                            f"(${tier_value:,.0f} CLV): {plan['headline']}.",
                    signal_type='churn_spike', data=plan)
            ctx.note_action()

        alert = tools.create_alert(
            ctx.company_id,
            f"{len(newly_at_risk)} customers newly crossed into high churn risk "
            f"(${total_value:,.0f} CLV at risk). See AI Agent recommendations.",
            severity='high' if total_value > 0 and len(newly_at_risk) >= min_new * 2 else 'medium',
            alert_type='churn',
        )
        ctx.log('act', "Raised an in-app alert summarizing the churn spike.",
                signal_type='churn_spike', data=alert)
        ctx.note_action()

    # --- Signal 2: portfolio CLV declining vs. the agent's own last snapshot ---

    def _check_clv_drop(self, ctx: ReasoningContext):
        cmp = tools.compare_period_over_period(ctx.state, metric='clv')
        if not cmp.get('available'):
            ctx.log('reason', f"CLV trend check: {cmp.get('reason', 'no data available')}.",
                    signal_type='clv_drop')
            return

        if not cmp.get('has_baseline'):
            ctx.log('reason', "CLV trend check: no prior snapshot to compare against yet - "
                               "recording this cycle's total CLV as the baseline.",
                    signal_type='clv_drop')
            memory.save_snapshot(ctx.company_id, 'total_clv', cmp['current_total_clv'])
            return

        change_pct = cmp['change_pct']
        ctx.log(
            'reason',
            f"Checked portfolio CLV vs. last recorded snapshot ({cmp['previous_recorded_at'][:10]}): "
            f"${cmp['current_total_clv']:,.0f} now vs ${cmp['previous_total_clv']:,.0f} then "
            f"({change_pct:+.1f}%).",
            signal_type='clv_drop',
            data={'change_pct': change_pct},
        )
        memory.save_snapshot(ctx.company_id, 'total_clv', cmp['current_total_clv'])

        drop_threshold = -config.THRESHOLDS['clv_drop_pct'] * 100
        if change_pct > drop_threshold:
            return

        if memory.has_recent_recommendation(ctx.company_id, 'clv_drop', config.THROTTLE_HOURS['clv_drop']):
            ctx.log('reason', "CLV drop still past threshold, but already flagged within the last "
                               f"{config.THROTTLE_HOURS['clv_drop']}h - not re-alerting yet.",
                    signal_type='clv_drop')
            return

        ctx.note_signal()
        segments = tools.segment_customers(ctx.state, criteria='value_tier')
        ctx.log('plan', "Plan: portfolio CLV dropped past threshold - segmenting by value tier to see "
                         "where the decline concentrates, then drafting a recovery plan.",
                signal_type='clv_drop', data=segments)

        plan = tools.generate_action_plan('clv_decline', context={
            'change_pct': change_pct,
            'current_total_clv': cmp['current_total_clv'],
            'previous_total_clv': cmp['previous_total_clv'],
            'segments': segments.get('segments', []),
        })
        body = self._format_plan(plan, [])
        tools.log_recommendation(
            ctx.company_id, body,
            title=f"{plan['headline']} ({change_pct:+.1f}%)",
            priority='high' if change_pct <= drop_threshold * 2 else 'medium',
            signal_type='clv_drop', cycle_id=ctx.cycle_id,
        )
        ctx.log('act', "Logged a CLV-decline recovery recommendation.", signal_type='clv_drop', data=plan)
        ctx.note_action()

        alert = tools.create_alert(
            ctx.company_id,
            f"Portfolio CLV declined {change_pct:.1f}% since the last check "
            f"(${cmp['previous_total_clv']:,.0f} -> ${cmp['current_total_clv']:,.0f}).",
            severity='high' if change_pct <= drop_threshold * 2 else 'medium',
            alert_type='clv',
        )
        ctx.log('act', "Raised an in-app alert for the CLV decline.", signal_type='clv_drop', data=alert)
        ctx.note_action()

    # --- Signal 3: forecast or actual revenue trending down ------------------

    def _check_revenue_dip(self, ctx: ReasoningContext):
        trend = tools.get_forecast_trend(ctx.state)
        mom = tools.compare_period_over_period(ctx.state, metric='sales')

        pieces = []
        if trend.get('available') and trend.get('growth_rate_pct') is not None:
            pieces.append(f"forecast growth {trend['growth_rate_pct']:+.1f}% over the horizon")
        if mom.get('available'):
            pieces.append(f"{mom['current_period']} vs {mom['previous_period']}: {mom['change_pct']:+.1f}%")

        if not pieces:
            ctx.log('reason', "Revenue trend check: no forecast or transaction trend data available.",
                    signal_type='revenue_dip')
            return

        ctx.log('reason', "Checked revenue trend - " + "; ".join(pieces) + ".",
                signal_type='revenue_dip',
                data={'forecast': trend, 'month_over_month': mom})

        forecast_fires = (trend.get('available') and trend.get('growth_rate_pct') is not None
                           and trend['growth_rate_pct'] <= config.THRESHOLDS['revenue_dip_growth_pct'])
        mom_fires = (mom.get('available')
                     and mom['change_pct'] <= config.THRESHOLDS['revenue_dip_mom_pct'])

        if not (forecast_fires or mom_fires):
            return

        if memory.has_recent_recommendation(ctx.company_id, 'revenue_dip', config.THROTTLE_HOURS['revenue_dip']):
            ctx.log('reason', "Revenue dip still past threshold, but already flagged within the last "
                               f"{config.THROTTLE_HOURS['revenue_dip']}h - not re-alerting yet.",
                    signal_type='revenue_dip')
            return

        ctx.note_signal()
        context = {'forecast': trend if trend.get('available') else None,
                   'month_over_month': mom if mom.get('available') else None}
        ctx.log('plan', "Plan: revenue dip confirmed - drafting a response plan and alerting now.",
                signal_type='revenue_dip')

        plan = tools.generate_action_plan('revenue_dip', context=context)
        body = self._format_plan(plan, [])
        worst_pct = min(
            trend['growth_rate_pct'] if forecast_fires else 0,
            mom['change_pct'] if mom_fires else 0,
        )
        tools.log_recommendation(
            ctx.company_id, body,
            title=f"{plan['headline']} ({worst_pct:+.1f}%)",
            priority='high' if worst_pct <= config.THRESHOLDS['revenue_dip_growth_pct'] * 1.5 else 'medium',
            signal_type='revenue_dip', cycle_id=ctx.cycle_id,
        )
        ctx.log('act', "Logged a revenue-dip response recommendation.", signal_type='revenue_dip', data=plan)
        ctx.note_action()

        alert = tools.create_alert(
            ctx.company_id,
            f"Revenue trend warning: {'; '.join(pieces)}.",
            severity='high', alert_type='sales',
        )
        ctx.log('act', "Raised an in-app alert for the revenue dip.", signal_type='revenue_dip', data=alert)
        ctx.note_action()

    # --- Signal 4: forecast doesn't look like the usual seasonal pattern -----

    def _check_seasonal_anomaly(self, ctx: ReasoningContext):
        trend = tools.get_forecast_trend(ctx.state)
        if not trend.get('available') or trend.get('volatility') is None:
            ctx.log('reason', "Seasonal anomaly check: no forecast volatility data available.",
                    signal_type='seasonal_anomaly')
            return

        volatility = trend['volatility']
        ctx.log('reason', f"Checked forecast volatility: {volatility:.2f} "
                           f"(coefficient of variation across the forecast horizon).",
                signal_type='seasonal_anomaly', data={'volatility': volatility})

        if volatility < config.THRESHOLDS['seasonal_volatility']:
            return

        if memory.has_recent_recommendation(ctx.company_id, 'seasonal_anomaly', config.THROTTLE_HOURS['seasonal_anomaly']):
            ctx.log('reason', "Forecast volatility still elevated, but already flagged within the last "
                               f"{config.THROTTLE_HOURS['seasonal_anomaly']}h - not re-alerting yet.",
                    signal_type='seasonal_anomaly')
            return

        ctx.note_signal()
        ctx.log('plan', "Plan: forecast volatility is unusually high - logging a low-priority "
                         "investigation item rather than a hard alert, since this is a softer signal.",
                signal_type='seasonal_anomaly')

        plan = tools.generate_action_plan('seasonal_anomaly', context={'volatility': volatility})
        body = self._format_plan(plan, [])
        tools.log_recommendation(
            ctx.company_id, body,
            title=f"{plan['headline']} (volatility {volatility:.2f})",
            priority='low', signal_type='seasonal_anomaly', cycle_id=ctx.cycle_id,
        )
        ctx.log('act', "Logged a seasonal-anomaly investigation recommendation.",
                signal_type='seasonal_anomaly', data=plan)
        ctx.note_action()

    # --- Shared formatting -----------------------------------------------------

    @staticmethod
    def _format_plan(plan, customers):
        lines = [plan['headline'], '']
        for i, step in enumerate(plan['steps'], 1):
            lines.append(f"{i}. {step}")
        if customers:
            lines.append('')
            lines.append('Affected customers:')
            for c in customers[:config.MAX_CUSTOMERS_IN_DETAIL]:
                lines.append(f"- {c.get('name', c.get('customer_id'))}: "
                              f"{c.get('churn_probability', 0):.0%} churn risk, ${c.get('clv', 0):,.0f} CLV")
            if len(customers) > config.MAX_CUSTOMERS_IN_DETAIL:
                lines.append(f"...and {len(customers) - config.MAX_CUSTOMERS_IN_DETAIL} more.")
        return '\n'.join(lines)
