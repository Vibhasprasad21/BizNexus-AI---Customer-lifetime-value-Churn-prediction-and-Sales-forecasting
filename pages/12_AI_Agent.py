import streamlit as st
from datetime import datetime

from src.auth.session import requires_auth, get_logged_in_user_email
from src.agent import config as agent_config
from src.agent import memory, scheduler
from src.agent.core import Agent


STEP_ICON = {'perceive': '👁️', 'reason': '🧠', 'plan': '🗺️', 'act': '⚡', 'observe': '🔁'}
STEP_LABEL = {'perceive': 'PERCEIVE', 'reason': 'REASON', 'plan': 'PLAN', 'act': 'ACT', 'observe': 'OBSERVE'}
PRIORITY_COLOR = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#27ae60', 'critical': '#c0392b'}


def _apply_styling():
    st.markdown("""
    <style>
    .main { background-color: #f8f9fa; color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .dashboard-card {
        background-color: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem; margin-bottom: 1.5rem;
    }
    .card-header {
        color: #2c3e50; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;
        padding-bottom: 0.5rem; border-bottom: 2px solid #f0f0f0;
    }
    .trail-step {
        display: flex; gap: 0.75rem; padding: 0.6rem 0; border-bottom: 1px solid #f4f4f4;
    }
    .trail-step .icon { font-size: 1.1rem; }
    .trail-step .badge {
        display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
        color: #4a90e2; background: #e9f2ff; border-radius: 6px; padding: 0.1rem 0.5rem; margin-right: 0.5rem;
    }
    .rec-card {
        background: #fbfbfc; border: 1px solid #eee; border-left: 4px solid #4a90e2;
        border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem;
    }
    .priority-chip {
        display: inline-block; border-radius: 10px; padding: 0.1rem 0.6rem;
        font-size: 0.75rem; font-weight: 700; color: white; margin-left: 0.5rem;
    }
    .backend-pill {
        display: inline-block; background: #eef6ff; color: #357bd8; border-radius: 14px;
        padding: 0.2rem 0.8rem; font-size: 0.8rem; font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


def _render_trail(events, empty_message):
    if not events:
        st.info(empty_message)
        return
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    for e in events:
        icon = STEP_ICON.get(e['step'], '•')
        label = STEP_LABEL.get(e['step'], e['step'].upper())
        ts = e['created_at'][:16].replace('T', ' ')
        st.markdown(
            f'<div class="trail-step"><span class="icon">{icon}</span>'
            f'<div><span class="badge">{label}</span>'
            f'<span style="color:#7f8c8d;font-size:0.75rem;">{ts}</span><br>'
            f'<span>{e["message"]}</span></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def _render_recommendations(company_id):
    pending = memory.get_recommendations(company_id, status='pending', limit=30)
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-header">📋 Pending Recommendations</h3>', unsafe_allow_html=True)

    if not pending:
        st.info("No pending recommendations. Run the agent to generate new ones.")
    for rec in pending:
        color = PRIORITY_COLOR.get(rec['priority'], '#7f8c8d')
        st.markdown(
            f'<div class="rec-card"><strong>{rec["title"]}</strong>'
            f'<span class="priority-chip" style="background:{color};">{rec["priority"].upper()}</span>'
            f'<div style="color:#7f8c8d;font-size:0.8rem;margin-top:0.3rem;">'
            f'{rec["created_at"][:16].replace("T", " ")} · signal: {rec.get("signal_type") or "n/a"}</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("View full recommendation"):
            st.markdown(rec['body'])
            if rec['target_customers']:
                st.caption(f"{len(rec['target_customers'])} customer(s) targeted")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", key=f"approve_{rec['id']}", use_container_width=True):
                    memory.decide_recommendation(rec['id'], 'approved', decided_by=get_logged_in_user_email())
                    st.rerun()
            with col2:
                if st.button("❌ Dismiss", key=f"dismiss_{rec['id']}", use_container_width=True):
                    memory.decide_recommendation(rec['id'], 'dismissed', decided_by=get_logged_in_user_email())
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def _render_activity_log(company_id):
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-header">🗂️ Agent Activity Log</h3>', unsafe_allow_html=True)

    cycles = memory.get_cycles(company_id, limit=15)
    if not cycles:
        st.info("No agent runs yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    for cycle in cycles:
        status_icon = '✅' if cycle['status'] == 'complete' else '⏳'
        started = cycle['started_at'][:16].replace('T', ' ')
        header = f"{status_icon} {started} · {cycle['trigger']} · {cycle.get('summary') or 'running...'}"
        with st.expander(header):
            st.markdown(f"<span class='backend-pill'>{cycle.get('reasoning_backend', 'rule_based')}</span>",
                        unsafe_allow_html=True)
            events = memory.get_events_for_cycle(cycle['id'])
            for e in events:
                icon = STEP_ICON.get(e['step'], '•')
                label = STEP_LABEL.get(e['step'], e['step'].upper())
                st.markdown(f"{icon} **{label}** — {e['message']}")

    decided = [r for r in memory.get_recommendations(company_id, limit=100) if r['status'] != 'pending']
    if decided:
        st.markdown("#### Past decisions & outcomes")
        for r in decided[:15]:
            outcome = r.get('outcome')
            outcome_text = f" · outcome: **{outcome.replace('_', ' ')}**" if outcome else " · outcome: not yet observed"
            st.markdown(f"- **{r['title']}** — {r['status']} by {r.get('decided_by') or 'unknown'}{outcome_text}")

    st.markdown('</div>', unsafe_allow_html=True)


def _render_scheduler_controls():
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-header">⏱️ Background Scheduling</h3>', unsafe_allow_html=True)
    st.caption(
        "Runs the agent automatically for every company with uploaded data, on a timer, "
        "using the `schedule` package - no need to keep this page open. Requires the "
        "Streamlit process to stay alive continuously; see the deployment note below."
    )

    running = scheduler.is_running()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        interval = st.number_input("Interval (minutes)", min_value=5, max_value=1440, value=60, step=5,
                                    disabled=running)
    with col2:
        if not running and st.button("▶️ Start", use_container_width=True):
            scheduler.start(interval_minutes=interval)
            st.rerun()
    with col3:
        if running and st.button("⏹️ Stop", use_container_width=True):
            scheduler.stop()
            st.rerun()

    st.markdown(f"**Status:** {'🟢 running' if running else '⚪ stopped'}")
    last = scheduler.last_run_summary()
    if last:
        st.caption(f"Last sweep: {datetime.fromtimestamp(last['ran_at']).strftime('%Y-%m-%d %H:%M')} "
                   f"across {len(last['companies'])} compan{'y' if len(last['companies']) == 1 else 'ies'}.")

    st.warning(
        "⚠️ **Streamlit Community Cloud reminder:** SQLite there lives on ephemeral storage - the app's "
        "container can be redeployed or put to sleep at any time, wiping the database (including this "
        "agent's memory: past cycles, recommendations, and outcomes) along with it. Background scheduling "
        "also stops the moment the process is recycled. For real persistence and always-on scheduling, "
        "self-host (a VM/container that stays up) or point the DB at durable storage."
    )
    st.markdown('</div>', unsafe_allow_html=True)


def _render_backend_note():
    backend = agent_config.REASONING_BACKEND
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-header">🧭 Reasoning Backend</h3>', unsafe_allow_html=True)
    st.markdown(f"<span class='backend-pill'>{backend}</span>", unsafe_allow_html=True)
    if backend == 'rule_based':
        st.caption(
            "100% local, deterministic, zero setup: a scored priority list of signals (churn spike, "
            "CLV drop, revenue dip, seasonal anomaly), each evaluated against a threshold and, if it "
            "fires, resolved by chaining several tool calls. It handles exactly the four signal types "
            "it was coded for, and won't generalize to a novel pattern nobody wrote a rule for. "
            "Set the `BIZNEXUS_AGENT_BACKEND=ollama` environment variable to swap in a local LLM "
            "(requires Ollama installed and running - see src/agent/reasoning/ollama_backend.py; "
            "not available on Streamlit Community Cloud)."
        )
    else:
        st.caption(
            "LLM-backed via a local Ollama server, reading the same tools as the rule-based engine. "
            "Can reason about combinations no explicit rule covers, at the cost of needing Ollama "
            "installed and running wherever this app is hosted - it will not work on Streamlit "
            "Community Cloud."
        )
    st.markdown('</div>', unsafe_allow_html=True)


@requires_auth
def main():
    st.set_page_config(page_title="BizNexus AI Agent", page_icon="🕵️", layout="wide")
    _apply_styling()

    st.title("🕵️ AI Agent")
    st.caption("Autonomously monitors your data, reasons about what matters, plans multi-step actions, "
               "and executes them - with a full, transparent trail of why.")

    company_id = st.session_state.get('company_id')
    if not company_id:
        st.error("No company context found. Please log in again.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        run_clicked = st.button("▶️ Run Agent Now", type="primary", use_container_width=True)
    with col2:
        st.caption("Perceives your current CLV/churn/forecast data, reasons about what's significant, "
                    "plans a response, and acts - in one pass.")

    if run_clicked:
        with st.spinner("Running perceive → reason → plan → act → observe..."):
            agent = Agent()
            result = agent.run_cycle(company_id, session_state=st.session_state, trigger='manual')
        st.session_state['_last_agent_result'] = result
        st.success(result['summary'])

    last_result = st.session_state.get('_last_agent_result')
    st.markdown("### Live Reasoning Trail")
    if last_result:
        st.caption(f"From the run above · perceived state source: **{last_result['state_source']}** · "
                   f"backend: **{last_result['backend']}**")
        _render_trail(last_result['trail'], "No events recorded for this run.")
    else:
        last_cycle = memory.get_last_cycle(company_id)
        if last_cycle:
            _render_trail(memory.get_events_for_cycle(last_cycle['id']),
                          "No events recorded for the last run.")
        else:
            st.info("The agent hasn't run yet. Click \"Run Agent Now\" to perceive your data and act on it.")

    _render_recommendations(company_id)
    _render_backend_note()
    _render_scheduler_controls()
    _render_activity_log(company_id)


if __name__ == "__main__":
    main()
