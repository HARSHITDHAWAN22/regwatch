"""
RegWatch dashboard - upload circulars, watch async processing, review
flagged impact assessments, and see live cost/cache metrics.

Run with: streamlit run frontend/streamlit_app.py
Expects the FastAPI backend running (default http://localhost:8000).
"""
import os
import time
import requests
import streamlit as st

try:
    import pandas as pd
except ImportError:
    pd = None

st.set_page_config(page_title="RegWatch", page_icon="🖋️", layout="wide")

# ============================================================
# THEME — "audit ledger": treats the dashboard like an official
# compliance register. Severity = ink stamps, not just colored
# text; data uses a monospace "register" typeface; the header
# reads like a circular's own letterhead.
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --ink: #1a2332;
    --ink-soft: #4a5568;
    --paper: #f0f1f5;
    --paper-card: #ffffff;
    --rule: #c9cdd6;
    --seal-urgent: #8b1e1e;
    --seal-action: #a9670c;
    --seal-info: #1f4e6b;
    --gold: #a9670c;
}

.stApp {
    background: var(--paper);
}

/* ---- Letterhead header ---- */
.rw-letterhead {
    border-bottom: 3px double var(--ink);
    padding-bottom: 14px;
    margin-bottom: 6px;
}
.rw-letterhead .rw-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--ink-soft);
    text-transform: uppercase;
    margin-bottom: 4px;
}
.rw-letterhead h1 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 700;
    font-size: 34px;
    color: var(--ink);
    margin: 0 0 6px 0;
    letter-spacing: 0.3px;
}
.rw-letterhead .rw-sub {
    font-family: 'Inter', sans-serif;
    font-size: 13.5px;
    color: var(--ink-soft);
}

/* ---- General typography ---- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; color: var(--ink) !important; }

/* ---- Main-area form widgets — force paper theme regardless of the
   browser/OS dark-mode setting, so they never mismatch the custom cards ---- */
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stTextArea"] textarea,
.stApp [data-testid="stNumberInput"] input {
    background: var(--paper-card) !important;
    color: var(--ink) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
}
.stApp [data-testid="stTextInput"] input:focus,
.stApp [data-testid="stTextArea"] textarea:focus {
    border-color: var(--seal-info) !important;
    box-shadow: 0 0 0 1px var(--seal-info) !important;
}
.stApp [data-testid="stSelectbox"] > div > div,
.stApp [data-testid="stFileUploaderDropzone"] {
    background: var(--paper-card) !important;
    color: var(--ink) !important;
    border: 1px solid var(--rule) !important;
}
.stApp [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
.stApp [data-testid="stFileUploaderDropzone"] small { color: var(--ink-soft) !important; }
.stApp [data-testid="stFileUploaderDropzone"] button {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: 1px solid var(--rule) !important;
}
.stApp label, .stApp .stMarkdown p, .stApp [data-testid="stWidgetLabel"] p {
    color: var(--ink) !important;
}
.stApp [data-testid="stCaptionContainer"] { color: var(--ink-soft) !important; }
.stApp [data-testid="stFileUploaderFile"] {
    background: var(--paper) !important;
    color: var(--ink) !important;
}

/* ---- Alert boxes (st.info / success / warning / error) — force readable
   contrast; these were unstyled and inherited near-invisible light text ---- */
.stApp [data-testid="stAlert"] {
    background: var(--paper-card) !important;
    border: 1px solid var(--rule) !important;
    border-left: 4px solid var(--seal-info) !important;
}
.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stAlert"] div,
.stApp [data-testid="stAlert"] span {
    color: var(--ink) !important;
}
.stApp [data-testid="stAlert"] code {
    background: var(--paper) !important;
    color: var(--seal-info) !important;
}
.stApp [data-testid="stNotificationContentSuccess"] { color: var(--ink) !important; }
.stApp [data-testid="stNotificationContentWarning"] { color: var(--ink) !important; }
.stApp [data-testid="stNotificationContentError"] { color: var(--ink) !important; }

/* ---- Sidebar as a case-file panel ---- */
section[data-testid="stSidebar"] {
    background: var(--ink);
    border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"] * { color: #e8e9ee !important; }
section[data-testid="stSidebar"] .rw-file-tab {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #9aa3b5 !important;
    border-bottom: 1px solid #37415a;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
section[data-testid="stSidebar"] input {
    background: #232d42 !important;
    color: #f0f1f5 !important;
    border: 1px solid #3a4560 !important;
}

/* ---- Tabs styled like ledger dividers ---- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid var(--ink); }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: #e4e6ec;
    border: 1px solid var(--rule);
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    color: var(--ink-soft);
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    background: var(--paper-card) !important;
    color: var(--ink) !important;
    font-weight: 600;
}

/* ---- Buttons ---- */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-radius: 3px;
    border: 1.5px solid var(--ink);
    background: var(--ink);
    color: #fff;
    padding: 6px 16px;
}
.stButton > button:hover { background: var(--paper-card); color: var(--ink); }
.stButton > button[kind="primary"] { background: var(--seal-urgent); border-color: var(--seal-urgent); }
.stButton > button[kind="primary"]:hover { background: var(--paper-card); color: var(--seal-urgent); }

/* ---- Stat / metric cards ---- */
.rw-stat-row { display: flex; gap: 14px; margin: 6px 0 18px 0; flex-wrap: wrap; }
.rw-stat {
    flex: 1; min-width: 160px;
    background: var(--paper-card);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--ink);
    border-radius: 4px;
    padding: 14px 16px;
}
.rw-stat .rw-stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--ink-soft);
}
.rw-stat .rw-stat-value {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 30px;
    font-weight: 700;
    color: var(--ink);
    margin-top: 2px;
}
.rw-stat .rw-stat-delta { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: var(--seal-action); }

/* ---- Impact "clause cards" ---- */
.rw-clause-card {
    background: var(--paper-card);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 4px 2px;
    margin-bottom: 2px;
}

/* ---- Ink stamp badges ---- */
.rw-stamp {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 3px 10px;
    border: 2px solid currentColor;
    border-radius: 3px;
    transform: rotate(-1.5deg);
}
.rw-stamp-urgent { color: var(--seal-urgent); }
.rw-stamp-action { color: var(--seal-action); }
.rw-stamp-info { color: var(--seal-info); }
.rw-flag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    color: var(--seal-urgent);
    border: 1px dashed var(--seal-urgent);
    padding: 2px 8px;
    border-radius: 3px;
    margin-left: 8px;
}

/* ---- Evidence highlight ---- */
.rw-evidence {
    background: #fdf3d8;
    padding: 1px 3px;
    border-bottom: 2px solid var(--gold);
    font-weight: 600;
}

/* ---- Score readout, ledger style ---- */
.rw-scoreline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: var(--ink-soft);
    border-top: 1px dotted var(--rule);
    border-bottom: 1px dotted var(--rule);
    padding: 6px 0;
    margin: 8px 0;
}

/* ---- Policy card ---- */
.rw-policy-card {
    background: var(--paper-card);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--seal-info);
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.rw-policy-card .rw-policy-name {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 700;
    font-size: 17px;
    color: var(--ink);
}
.rw-policy-card .rw-policy-owner {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--seal-info);
}
.rw-policy-card .rw-policy-desc {
    font-size: 13.5px;
    color: var(--ink-soft);
    margin-top: 4px;
}

/* ---- Expander header cleanup ---- */
.streamlit-expanderHeader { font-family: 'Inter', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

API_BASE = st.sidebar.text_input("Backend URL", value=os.environ.get("BACKEND_URL", "http://localhost:8000"))

st.markdown("""
<div class="rw-letterhead">
    <div class="rw-kicker">Auditable Regulatory Impact Register</div>
    <h1>🖋️ RegWatch</h1>
    <div class="rw-sub">Hybrid retrieval → reranking → LLM impact reasoning → self-consistency verification. Every finding traceable to a source clause.</div>
</div>
""", unsafe_allow_html=True)

# ---------- Auth ----------
if "token" not in st.session_state:
    st.session_state.token = None

with st.sidebar:
    st.markdown('<div class="rw-file-tab">Case File — Access</div>', unsafe_allow_html=True)
    auth_slot = st.container()
    with auth_slot:
        if st.session_state.token is None:
            email = st.text_input("Email", value="admin@regwatch.demo", key="login_email")
            password = st.text_input("Password", value="admin123", type="password", key="login_password")
            if st.button("Log in", key="login_submit_btn"):
                try:
                    r = requests.post(f"{API_BASE}/auth/login", json={"email": email, "password": password})
                    if r.status_code == 200:
                        st.session_state.token = r.json()["access_token"]
                        st.rerun()
                    else:
                        st.error(f"Login failed: {r.json().get('detail')}")
                except Exception as e:
                    st.error(f"Could not reach backend: {e}")
        else:
            st.success("Logged in")
            if st.button("Log out", key="logout_btn"):
                st.session_state.token = None
                st.rerun()

if st.session_state.token is None:
    st.info("Log in from the sidebar to use RegWatch. Seed a demo admin with `python -m scripts.seed`.")
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}

tab_upload, tab_impacts, tab_policies, tab_metrics = st.tabs(
    ["Upload Circular", "Impact Assessments", "Policy Registry", "Metrics & Cost"]
)

# ---------- Upload ----------
with tab_upload:
    st.subheader("File a new circular")
    st.caption("Try the sample circulars in data/sample_circulars/ to see the pipeline run end-to-end.")

    title = st.text_input("Circular title")
    circular_number = st.text_input("Circular number (optional)")
    uploaded_file = st.file_uploader("Circular file", type=["pdf", "txt"])

    if st.button("Ingest & Assess Impact", type="primary"):
        if not uploaded_file or not title:
            st.warning("Please provide a title and a file.")
        else:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {"title": title, "circular_number": circular_number}
            r = requests.post(f"{API_BASE}/circulars/upload", headers=headers, files=files, data=data)
            if r.status_code == 200:
                job_id = r.json()["job_id"]
                st.info(f"Job queued: `{job_id}` — processing runs in the background.")

                progress = st.empty()
                status = {"status": "processing"}
                for _ in range(60):
                    try:
                        status_r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=10)
                        status = status_r.json()
                        progress.write(status)
                        if status["status"] in ("completed", "failed"):
                            break
                    except Exception:
                        progress.write({"status": "processing", "note": "waiting on slow free-tier response..."})
                    time.sleep(3)

                if status["status"] == "completed":
                    st.success(f"Done — {status.get('impacts_found', 0)} impact(s) found. Check the Impact Assessments tab.")
                elif status["status"] == "failed":
                    st.error(f"Processing failed: {status.get('error')}")
            else:
                st.error(r.json())

# ---------- Impacts ----------
with tab_impacts:
    st.subheader("Impact Assessments")
    col1, col2 = st.columns(2)
    severity_filter = col1.selectbox("Filter by severity", ["All", "info", "action_required", "urgent"])
    flagged_only = col2.checkbox("Flagged for review only")

    params = {}
    if severity_filter != "All":
        params["severity"] = severity_filter
    if flagged_only:
        params["flagged_only"] = True

    r = requests.get(f"{API_BASE}/impacts", params=params)
    impacts = r.json() if r.status_code == 200 else []

    if not impacts:
        st.info("No impact assessments yet — upload a circular first.")

    stamp_class = {"urgent": "rw-stamp-urgent", "action_required": "rw-stamp-action", "info": "rw-stamp-info"}
    stamp_label = {"urgent": "Urgent", "action_required": "Action Req.", "info": "Info"}

    for imp in impacts:
        sev = imp["severity"]
        badge_html = f'<span class="rw-stamp {stamp_class.get(sev, "rw-stamp-info")}">{stamp_label.get(sev, sev)}</span>'
        flag_html = '<span class="rw-flag">⚑ Flagged for review</span>' if imp["is_flagged_for_review"] else ""

        with st.expander(f"{stamp_label.get(sev, sev)} — {imp['reasoning'][:90]}..."):
            st.markdown(f'<div class="rw-clause-card">{badge_html}{flag_html}</div>', unsafe_allow_html=True)
            st.write("")

            cited = imp["cited_text"]
            span_start, span_end = imp.get("span_start"), imp.get("span_end")
            if span_start is not None and span_end is not None and 0 <= span_start < span_end <= len(cited):
                before, evidence, after = cited[:span_start], cited[span_start:span_end], cited[span_end:]
                st.markdown("**Cited clause** _(highlighted = exact evidence used)_:")
                st.markdown(
                    f'<div style="padding:10px 14px;border-left:3px solid #c9cdd6;font-size:14px;">'
                    f'{before}<span class="rw-evidence">{evidence}</span>{after}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**Cited clause:**\n> {cited}")

            st.markdown(f"**Reasoning:** {imp['reasoning']}")

            st.markdown(
                f'<div class="rw-scoreline">RETRIEVAL {imp["retrieval_score"]:.3f}'
                f'&nbsp;&nbsp;·&nbsp;&nbsp;VERIFICATION {imp.get("verification_score", "—")}/5'
                f'&nbsp;&nbsp;·&nbsp;&nbsp;STATUS {imp["review_status"].upper()}</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            if c1.button("✅ Confirm", key=f"confirm-{imp['id']}"):
                requests.post(f"{API_BASE}/impacts/{imp['id']}/review", headers=headers,
                              json={"review_status": "confirmed"})
                st.rerun()
            if c2.button("❌ Reject", key=f"reject-{imp['id']}"):
                requests.post(f"{API_BASE}/impacts/{imp['id']}/review", headers=headers,
                              json={"review_status": "rejected"})
                st.rerun()
            note = c3.text_input("Correction note", key=f"note-{imp['id']}", label_visibility="collapsed",
                                  placeholder="Optional correction...")
            if note and c3.button("Submit correction", key=f"correct-{imp['id']}"):
                requests.post(f"{API_BASE}/impacts/{imp['id']}/review", headers=headers,
                              json={"review_status": "corrected", "correction_note": note})
                st.rerun()

# ---------- Policy Registry ----------
with tab_policies:
    st.subheader("Policy Registry")
    r = requests.get(f"{API_BASE}/policies")
    policies = r.json() if r.status_code == 200 else []

    for p in policies:
        st.markdown(
            f'<div class="rw-policy-card">'
            f'<div class="rw-policy-name">{p["name"]}</div>'
            f'<div class="rw-policy-owner">Owner — {p.get("owner_team") or "Unassigned"}</div>'
            f'<div class="rw-policy-desc">{p["description"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("➕ Add new policy"):
        name = st.text_input("Policy name")
        desc = st.text_area("Description")
        owner = st.text_input("Owner team")
        if st.button("Add policy"):
            r = requests.post(f"{API_BASE}/policies", headers=headers,
                               json={"name": name, "description": desc, "owner_team": owner})
            if r.status_code == 200:
                st.success("Policy added.")
                st.rerun()
            else:
                st.error(r.json())

# ---------- Metrics ----------
with tab_metrics:
    st.subheader("Observability")
    r = requests.get(f"{API_BASE}/metrics")
    if r.status_code == 200:
        m = r.json()
        cache = m["cache"]
        backend_label = {"redis": "Redis", "in-memory-fallback": "in-memory fallback"}.get(
            cache.get("backend"), cache.get("backend", "unknown")
        )

        st.markdown(f"""
        <div class="rw-stat-row">
            <div class="rw-stat">
                <div class="rw-stat-label">Total Assessments</div>
                <div class="rw-stat-value">{m['total_assessments']}</div>
            </div>
            <div class="rw-stat">
                <div class="rw-stat-label">Flagged for Review</div>
                <div class="rw-stat-value">{m['flagged_for_review']}</div>
                <div class="rw-stat-delta">{m['flagged_rate']*100:.1f}% of total</div>
            </div>
            <div class="rw-stat">
                <div class="rw-stat-label">Total Tokens Used</div>
                <div class="rw-stat-value">{m['total_tokens_used']:,}</div>
            </div>
            <div class="rw-stat">
                <div class="rw-stat-label">Est. Cost (USD)</div>
                <div class="rw-stat-value">${m['estimated_cost_usd']}</div>
            </div>
            <div class="rw-stat">
                <div class="rw-stat-label">Cache ({backend_label})</div>
                <div class="rw-stat-value">{cache['size']}</div>
                <div class="rw-stat-delta">of {cache['max_size']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Severity distribution chart, if we have impacts and pandas is available
        if pd is not None:
            r_imp = requests.get(f"{API_BASE}/impacts")
            if r_imp.status_code == 200 and r_imp.json():
                impacts_df = pd.DataFrame(r_imp.json())
                counts = impacts_df["severity"].value_counts().reindex(
                    ["urgent", "action_required", "info"]
                ).fillna(0)
                st.markdown("**Severity distribution**")
                st.bar_chart(counts, color="#a9670c")

        st.divider()
        st.subheader("Comparison by prompt/pipeline version")
        st.caption(
            "Every assessment is stamped with the prompt_version/pipeline_version active when it "
            "was created. After you edit a prompt and bump PROMPT_VERSION in .env, this table lets "
            "you compare outcomes across versions instead of only seeing a hidden-regression aggregate."
        )
        r2 = requests.get(f"{API_BASE}/metrics/by-version")
        if r2.status_code == 200 and r2.json():
            st.table(r2.json())
        else:
            st.info("No version comparison data yet.")
    else:
        st.error("Could not load metrics.")
