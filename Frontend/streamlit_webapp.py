import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZerveChurn · Churn Predictor",
    page_icon="🔮",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f0f11;
    color: #f0f0f5;
}
[data-testid="stSidebar"] {
    background-color: #16161a;
    border-right: 1px solid #2a2a32;
}

/* ── Headings ── */
h1 { color: #ffffff !important; font-size: 1.9rem !important; font-weight: 700 !important; }
h2, h3 { color: #e2e2ea !important; }
label, p, li, span { color: #c4c4d0 !important; }

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #1e1e26 !important;
    border: 1px solid #2e2e3a !important;
    color: #f0f0f5 !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.25) !important;
}

/* ── Primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #7c6af7, #a78bfa) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1.2rem !important;
    transition: opacity 0.2s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover { opacity: 0.88 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background-color: #1e1e26 !important;
    border: 1.5px dashed #3a3a4a !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: #2a2a36 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #1a1a22 !important;
    border: 1px solid #2a2a36 !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: #8888a0 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.7rem !important; font-weight: 700 !important; }

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Expander ── */
details summary { color: #a0a0bc !important; }
details[open] summary { color: #d0d0e8 !important; }
[data-testid="stExpander"] { background-color: #16161e !important; border: 1px solid #26262e !important; border-radius: 10px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; }

/* ── Sidebar labels ── */
.stSidebar label { color: #b0b0c8 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] { color: #8888a0 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #a78bfa !important; border-bottom-color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)

# ── Zerve palette ─────────────────────────────────────────────────────────────
BG       = "#1D1D20"
TXT      = "#fbfbff"
SEC      = "#909094"
C_RED    = "#FF9F9B"
C_ORANGE = "#FFB482"
C_GREEN  = "#8DE5A1"
C_BLUE   = "#A1C9F4"
C_PURPLE = "#a78bfa"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1a1a22",
    font=dict(color=TXT, family="Inter, sans-serif"),
    title_font=dict(color=TXT, size=16),
    xaxis=dict(gridcolor="#2a2a36", linecolor=SEC, tickcolor=SEC, zerolinecolor="#2a2a36"),
    yaxis=dict(gridcolor="#2a2a36", linecolor=SEC, tickcolor=SEC, zerolinecolor="#2a2a36"),
    legend=dict(bgcolor="rgba(26,26,34,0.8)", bordercolor="#2a2a36", font=dict(color=TXT)),
    margin=dict(t=60, b=50, l=55, r=30),
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_models(pkl_bytes: bytes) -> dict:
    return pickle.loads(pkl_bytes)


@st.cache_data(show_spinner="Running EDA pipeline…")
def run_eda_pipeline(csv_bytes: bytes) -> dict:
    """Reproduce the full EDA pipeline from the notebook."""
    import io, warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Parse timestamp — CSV stores tz-aware strings (e.g. 2025-09-01 16:36:11+00:00)
    # utc=True normalises all offsets to UTC then we strip tz for plain arithmetic
    if "timestamp" in df.columns:
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
              .dt.tz_localize(None)          # drop UTC tag → naive datetime
        )

    # ── 1. Dataset dimensions ────────────────────────────────────────────────
    n_rows, n_cols = df.shape
    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_counts = {str(k): v for k, v in dtype_counts.items()}

    missing_stats = df.isnull().sum()
    missing_pct   = (missing_stats / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing_stats, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)

    n_unique_events = df["event"].nunique() if "event" in df.columns else None
    n_unique_users  = df["distinct_id"].nunique() if "distinct_id" in df.columns else None
    top_events      = df["event"].value_counts().head(15) if "event" in df.columns else None
    memory_mb       = df.memory_usage(deep=True).sum() / 1024 ** 2

    # ── 2. User-level base ───────────────────────────────────────────────────
    user_base = df.groupby("distinct_id").agg(
        first_activity=("timestamp", "min"),
        last_activity=("timestamp", "max"),
        total_events=("timestamp", "count"),
        unique_event_types=("event", "nunique"),
    ).reset_index()
    user_base.rename(columns={"distinct_id": "user_id"}, inplace=True)
    # Already naive datetimes after the parse above — no tz conversion needed
    user_base["first_activity"] = pd.to_datetime(user_base["first_activity"])
    user_base["last_activity"]  = pd.to_datetime(user_base["last_activity"])
    current_date = user_base["last_activity"].max()

    user_base["days_since_first"] = (current_date - user_base["first_activity"]).dt.total_seconds() / 86400
    user_base["days_since_last"]  = (current_date - user_base["last_activity"]).dt.total_seconds()  / 86400
    user_base["tenure_days"]      = (user_base["last_activity"] - user_base["first_activity"]).dt.total_seconds() / 86400

    # ── 3. Primary success metrics ───────────────────────────────────────────
    user_base["active_last_30d"]       = user_base["days_since_last"] <= 30
    user_base["eligible_for_retention"] = user_base["days_since_first"] >= 90
    user_base["long_term_retention"]   = user_base["active_last_30d"] & user_base["eligible_for_retention"]

    credit_events = ["credit_balance_updated", "credits_purchased", "credit_usage_tracked"]
    paid_users = df[df["event"].isin(credit_events)]["distinct_id"].unique()
    user_base["is_paid_user"] = user_base["user_id"].isin(paid_users)

    deployment_events = [
        "api_deployed", "endpoint_deployed", "model_deployed",
        "sagemaker_endpoint_deployed", "api_route_created",
        "deployment_successful", "canvas_published",
    ]
    deployed_users = df[df["event"].isin(deployment_events)]["distinct_id"].unique()
    user_base["has_deployment"] = user_base["user_id"].isin(deployed_users)

    collab_events = [
        "canvas_shared", "canvas_shared_with_user", "share_link_created",
        "comment_added", "comment_replied", "user_invited",
        "workspace_invite_sent", "mention_notification",
    ]
    collab_counts = df[df["event"].isin(collab_events)].groupby("distinct_id").size().reset_index(name="collab_count")
    user_base = user_base.merge(collab_counts, left_on="user_id", right_on="distinct_id", how="left")
    user_base["collab_count"] = user_base["collab_count"].fillna(0)
    user_base["collaboration_success"] = user_base["collab_count"] >= 3

    eligible60 = user_base[user_base["days_since_first"] >= 60].copy()
    user_base["is_power_user"] = False
    if len(eligible60) > 0:
        eligible60["engagement_rate"] = eligible60["total_events"] / eligible60["tenure_days"].clip(lower=1)
        threshold = eligible60["engagement_rate"].quantile(0.75)
        eligible60["is_power_user"] = eligible60["engagement_rate"] >= threshold
        user_base = user_base.merge(eligible60[["user_id", "is_power_user"]], on="user_id", how="left", suffixes=("", "_new"))
        user_base["is_power_user"] = user_base["is_power_user_new"].fillna(user_base["is_power_user"])
        user_base.drop(columns=["is_power_user_new"], inplace=True)

    # ── 4. Composite success score ───────────────────────────────────────────
    weights = {"long_term_retention": 0.30, "is_paid_user": 0.25, "has_deployment": 0.20,
               "is_power_user": 0.15, "collaboration_success": 0.10}

    def _score(row):
        return sum(weights[m] * 100 for m in weights if row[m])

    user_base["success_score"] = user_base.apply(_score, axis=1)

    def _label(s):
        return "Successful" if s > 60 else ("Moderate" if s > 30 else "Failed")

    user_base["success_label"] = user_base["success_score"].apply(_label)

    # Alternative labeling
    def _alt_score(row):
        s = 0
        if row["total_events"] >= 100:   s += 25
        elif row["total_events"] >= 10:  s += 15
        elif row["total_events"] >= 5:   s += 10
        if row["long_term_retention"]:   s += 30
        elif row["days_since_first"] >= 30: s += 15
        elif row["days_since_first"] >= 7:  s += 5
        if row["is_power_user"]:         s += 20
        if row["tenure_days"] >= 30:     s += 15
        elif row["tenure_days"] >= 7:    s += 10
        return s

    user_base["alternative_score"] = user_base.apply(_alt_score, axis=1)

    def _alt_label(s):
        return "High Value" if s >= 60 else ("Growing" if s >= 30 else "Early/Churned")

    user_base["alternative_label"] = user_base["alternative_score"].apply(_alt_label)

    # ── 5. Survival / churn analysis ─────────────────────────────────────────
    survival = user_base.copy()
    survival["churned"]       = (survival["days_since_last"] > 30).astype(int)
    survival["time_to_event"] = survival["tenure_days"].clip(lower=1)
    survival.loc[survival["churned"] == 0, "time_to_event"] = survival.loc[survival["churned"] == 0, "days_since_first"]
    survival = survival[survival["time_to_event"] > 0].copy()

    def _risk_segment(row):
        s = 0
        if row["total_events"] < 10:       s += 2
        if row["days_since_last"] > 7:     s += 2
        if row["tenure_days"] < 7:         s += 1
        if not row["is_paid_user"]:        s += 1
        if not row["has_deployment"]:      s += 1
        if not row["collaboration_success"]: s += 1
        return "High Risk" if s >= 5 else ("Medium Risk" if s >= 3 else "Low Risk")

    survival["risk_segment"] = survival.apply(_risk_segment, axis=1)
    survival["engagement_level"] = "Low"
    survival.loc[survival["total_events"] >= 50, "engagement_level"] = "High"
    survival.loc[(survival["total_events"] >= 10) & (survival["total_events"] < 50), "engagement_level"] = "Medium"
    survival["deployment_status"] = survival["has_deployment"].map({True: "Has Deployment", False: "No Deployment"})

    # ── 6. Browser/OS distributions ─────────────────────────────────────────
    browser_dist = None
    os_dist      = None
    country_dist = None
    if "prop_$browser" in df.columns:
        browser_dist = df["prop_$browser"].value_counts().head(10)
    if "prop_$os" in df.columns:
        os_dist = df["prop_$os"].value_counts().head(10)
    if "prop_$geoip_country_code" in df.columns:
        country_dist = df["prop_$geoip_country_code"].value_counts().head(15)

    # ── 7. Activity over time ────────────────────────────────────────────────
    activity_over_time = None
    if "timestamp" in df.columns:
        tmp = df.copy()
        tmp["week"] = tmp["timestamp"].dt.to_period("W").dt.start_time
        activity_over_time = tmp.groupby("week").size().reset_index(name="event_count")

    return dict(
        df=df,
        user_base=user_base,
        survival=survival,
        n_rows=n_rows, n_cols=n_cols,
        dtype_counts=dtype_counts,
        missing_df=missing_df,
        memory_mb=memory_mb,
        n_unique_events=n_unique_events,
        n_unique_users=n_unique_users,
        top_events=top_events,
        current_date=current_date,
        browser_dist=browser_dist,
        os_dist=os_dist,
        country_dist=country_dist,
        activity_over_time=activity_over_time,
    )


MODEL_DISPLAY_NAMES = {
    "best_model":           "Best Model (auto-selected)",
    "random_forest":        "Random Forest",
    "gradient_boosting":    "Gradient Boosting",
    "adaboost":             "AdaBoost",
    "logistic_regression":  "Logistic Regression",
    "voting_ensemble":      "Voting Ensemble",
    "stacking_ensemble":    "Stacking Ensemble",
}


def compute_churn_window(churn_score: float, days_since_last: float) -> tuple[str, str]:
    if churn_score >= 70 and days_since_last >= 14:
        return "Within 30 days", "#ef4444"
    elif churn_score >= 50 and days_since_last >= 7:
        return "Within 60 days", "#f97316"
    elif churn_score >= 30:
        return "Within 90 days", "#eab308"
    else:
        return "Beyond 90 days", "#22c55e"


def risk_level(churn_score: float) -> tuple[str, str, str]:
    if churn_score >= 70:
        return "High Risk", "#ef4444", "🔴"
    elif churn_score >= 40:
        return "Moderate Risk", "#f97316", "🟠"
    else:
        return "Low Risk", "#22c55e", "🟢"


def score_gauge_html(score: float, color: str) -> str:
    pct = score / 100
    r = 70
    circumference = 3.14159 * r
    dash = pct * circumference
    gap  = circumference - dash
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin:0.4rem 0">
      <svg width="180" height="110" viewBox="0 0 180 110">
        <path d="M 20 90 A 70 70 0 0 1 160 90"
              fill="none" stroke="#2a2a36" stroke-width="14" stroke-linecap="round"/>
        <path d="M 20 90 A 70 70 0 0 1 160 90"
              fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
              stroke-dasharray="{dash:.1f} {gap:.1f}"
              style="transition:stroke-dasharray 0.6s ease"/>
        <text x="90" y="88" text-anchor="middle"
              font-size="28" font-weight="700" fill="#ffffff">{score:.0f}</text>
        <text x="90" y="104" text-anchor="middle"
              font-size="11" fill="#8888a0">CHURN SCORE</text>
      </svg>
    </div>
    """


def km_estimate(time_arr, event_arr):
    """Manual Kaplan-Meier estimator."""
    df = pd.DataFrame({"time": time_arr, "event": event_arr}).sort_values("time")
    times, surv, n = [], [], len(df)
    cum = 1.0
    for t in df["time"].unique():
        at = df[df["time"] == t]
        n_ev = at["event"].sum()
        if n > 0 and n_ev > 0:
            cum *= (n - n_ev) / n
        times.append(t)
        surv.append(cum)
        n -= len(at)
    return np.array(times), np.array(surv)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔮 ZerveChurn")
    st.caption("Behavioral churn intelligence")
    st.divider()

    st.markdown("**1 · Upload model file**")
    pkl_file = st.file_uploader(
        "ensemble_models.pkl", type=["pkl"], label_visibility="collapsed"
    )

    models_data = None
    best_name   = None

    if pkl_file:
        try:
            models_data = load_models(pkl_file.read())
            best_name   = models_data.get("best_model_name", "N/A")
            st.success(f"Loaded · Best: **{best_name}**")
        except Exception as e:
            st.error(f"Could not load pkl: {e}")

    st.divider()
    st.markdown("**2 · Choose model**")
    model_keys   = list(MODEL_DISPLAY_NAMES.keys())
    selected_key = st.selectbox(
        "Model", model_keys,
        format_func=lambda k: MODEL_DISPLAY_NAMES[k],
        label_visibility="collapsed",
    )

    if models_data:
        st.divider()
        st.markdown("**Validation scores**")
        val_results = models_data.get("validation_results")
        if val_results is not None:
            display_df = val_results[["Model", "Accuracy", "F1 Score"]].copy()
            display_df["Accuracy"] = display_df["Accuracy"].map("{:.3f}".format)
            display_df["F1 Score"] = display_df["F1 Score"].map("{:.3f}".format)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Built on the ZerveChurn GNN pipeline · ensemble_models.pkl + user_retention.csv")


# ── Tab layout ────────────────────────────────────────────────────────────────
tab_predict, tab_eda = st.tabs(["🔮 Predictor", "📊 EDA & Insights"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("## Customer Churn Predictor")
    st.caption(
        "Enter a user's behavioral metrics to estimate their churn risk "
        "and predict when they are most likely to disengage."
    )
    st.divider()

    st.markdown("#### User activity profile")
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        total_events = st.number_input(
            "Total events", min_value=0, max_value=100_000, value=50, step=1,
            help="Total number of tracked events for this user across their lifetime.",
        )
        days_since_first = st.number_input(
            "Days since first activity", min_value=0, max_value=3_650, value=45, step=1,
            help="How many days ago this user first appeared in the system.",
        )

    with col2:
        tenure_days = st.number_input(
            "Tenure (days)", min_value=0, max_value=3_650, value=30, step=1,
            help="Span between a user's first and last recorded activity in days.",
        )
        days_since_last = st.number_input(
            "Days since last activity", min_value=0, max_value=3_650, value=7, step=1,
            help="How many days have passed since this user was last active.",
        )

    if days_since_last > days_since_first:
        st.warning("Days since last activity cannot exceed days since first activity.")
    if tenure_days > days_since_first:
        st.warning("Tenure should not exceed days since first activity.")

    st.divider()
    predict_btn = st.button("Run prediction", type="primary", use_container_width=True)

    if predict_btn:
        if models_data is None:
            st.error("Upload your **ensemble_models.pkl** in the sidebar to run a prediction.")
            st.stop()

        try:
            model  = models_data[selected_key]
            scaler = models_data["scaler"]
        except KeyError as e:
            st.error(f"Could not find key `{e}` in the pkl file.")
            st.stop()

        X_raw = np.array([[total_events, tenure_days, days_since_first, days_since_last]], dtype=float)
        try:
            X_scaled = scaler.transform(X_raw)
            proba    = model.predict_proba(X_scaled)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        success_proba = float(proba[1])
        churn_score   = round((1 - success_proba) * 100, 1)
        window_label, window_color = compute_churn_window(churn_score, days_since_last)
        risk_label, risk_color, risk_emoji = risk_level(churn_score)

        st.markdown("#### Prediction results")
        r1, r2, r3 = st.columns([1.3, 1, 1], gap="medium")
        with r1:
            st.markdown(score_gauge_html(churn_score, risk_color), unsafe_allow_html=True)
        with r2:
            st.metric("Churn window", window_label)
            st.metric("Retention score", f"{100 - churn_score:.0f} / 100")
        with r3:
            st.metric("Risk tier", f"{risk_emoji} {risk_label}")
            model_used = best_name if selected_key == "best_model" else MODEL_DISPLAY_NAMES[selected_key]
            st.metric("Model used", model_used or selected_key)

        st.divider()

        window_map = {
            "Within 30 days": (1, "#ef4444"), "Within 60 days": (2, "#f97316"),
            "Within 90 days": (3, "#eab308"), "Beyond 90 days": (4, "#22c55e"),
        }
        stage, _ = window_map[window_label]
        stage_labels = ["30 days", "60 days", "90 days", "90+ days"]
        bar_html = '<div style="display:flex;gap:6px;margin:0.3rem 0 0.8rem">'
        for i, lbl in enumerate(stage_labels, start=1):
            active, past = i == stage, i < stage
            bg = window_color if active else ("#2a2a36" if past else "#1e1e28")
            tc = "#ffffff" if active else ("#666678" if past else "#44444c")
            fw = "700" if active else "400"
            bar_html += (
                f'<div style="flex:1;background:{bg};border-radius:8px;'
                f'padding:6px 0;text-align:center;font-size:0.78rem;'
                f'color:{tc};font-weight:{fw}">{lbl}</div>'
            )
        bar_html += "</div>"
        st.markdown("**Churn timeline**")
        st.markdown(bar_html, unsafe_allow_html=True)

        if churn_score >= 70:
            st.error(
                f"**High churn risk ({churn_score}/100).** This user shows strong disengagement "
                f"signals. With {days_since_last} days of inactivity and a churn window of "
                f"**{window_label.lower()}**, immediate intervention is recommended — "
                "consider a personal outreach, re-activation campaign, or in-app prompt."
            )
        elif churn_score >= 40:
            st.warning(
                f"**Moderate churn risk ({churn_score}/100).** The user is at risk but not "
                f"critical. They have been inactive for {days_since_last} days with a churn "
                f"window of **{window_label.lower()}**. Proactive nudges or feature "
                "highlights could extend engagement before disengagement accelerates."
            )
        else:
            st.success(
                f"**Low churn risk ({churn_score}/100).** This user shows healthy behavioral "
                f"signals. With a retention score of {100 - churn_score:.0f}/100 and a churn "
                f"window of **{window_label.lower()}**, no immediate action is needed. "
                "Monitor for any sudden drop in activity."
            )

        with st.expander("Input summary", expanded=False):
            recap_df = pd.DataFrame({
                "Feature": ["total_events", "tenure_days", "days_since_first", "days_since_last"],
                "Value":   [total_events, tenure_days, days_since_first, days_since_last],
                "Description": [
                    "Lifetime event count", "First-to-last activity span (days)",
                    "Days since account was first active", "Days since last recorded activity",
                ],
            })
            st.dataframe(recap_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Raw prediction probabilities: "
                f"P(High Value) = {success_proba:.4f} · "
                f"P(Not High Value) = {1 - success_proba:.4f}"
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA & INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("## EDA & Insights")
    st.caption("Upload your dataset to explore behavioral patterns, survival curves, and success segmentation.")

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1.5px dashed #7c6af7;
        border-radius: 14px;
        padding: 1.6rem 2rem 1rem;
        margin-bottom: 1.2rem;
    ">
        <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.4rem">
            <span style="font-size:1.5rem">📂</span>
            <span style="color:#e2e2ea;font-size:1.05rem;font-weight:600">Load Dataset</span>
        </div>
        <p style="color:#8888a0;font-size:0.85rem;margin:0 0 0.8rem">
            Drop in <code style="background:#2a2a36;padding:2px 6px;border-radius:4px;color:#a78bfa">user_retention.csv</code>
            to unlock all six EDA sections — dataset overview, behavioral metrics, success scoring,
            survival curves, cohort analysis, and the user profile explorer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    csv_file = st.file_uploader(
        "Choose user_retention.csv",
        type=["csv"],
        label_visibility="collapsed",
        help="The raw event-level CSV exported from the ZerveChurn pipeline.",
    )

    if csv_file is None:
        st.markdown("""
        <div style="
            display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin-top:1.2rem
        ">
        """ + "".join([
            f"""<div style="background:#1a1a22;border:1px solid #2a2a36;border-radius:10px;
                           padding:1rem;text-align:center">
                <div style="font-size:1.6rem">{icon}</div>
                <div style="color:#e2e2ea;font-weight:600;margin:0.3rem 0;font-size:0.9rem">{title}</div>
                <div style="color:#8888a0;font-size:0.78rem">{desc}</div>
            </div>"""
            for icon, title, desc in [
                ("🗂️", "Dataset Overview", "Rows, columns, missing values & event frequency"),
                ("📈", "Survival Curves", "Kaplan-Meier curves by risk, engagement & deployment"),
                ("🏷️", "Success Scoring", "Composite scores, labels & cohort heatmaps"),
                ("⏱️", "Tenure Cohorts", "Churn rate & activity by user lifecycle stage"),
                ("🌍", "Device & Geo", "Browser, OS and country breakdowns"),
                ("🔍", "User Explorer", "Browse the full user-level summary table"),
            ]
        ]) + "</div>", unsafe_allow_html=True)
        st.stop()

    eda = run_eda_pipeline(csv_file.read())
    user_base = eda["user_base"]
    survival  = eda["survival"]

    # ── Section 1: Dataset overview ──────────────────────────────────────────
    st.markdown("### 1 · Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total rows",     f"{eda['n_rows']:,}")
    m2.metric("Columns",        f"{eda['n_cols']}")
    m3.metric("Unique users",   f"{eda['n_unique_users']:,}")
    m4.metric("Unique events",  f"{eda['n_unique_events']:,}")
    m5.metric("Dataset size",   f"{eda['memory_mb']:.1f} MB")

    col_a, col_b = st.columns(2, gap="medium")

    # Column type breakdown
    with col_a:
        dtype_df = pd.DataFrame.from_dict(eda["dtype_counts"], orient="index", columns=["count"]).reset_index()
        dtype_df.columns = ["dtype", "count"]
        fig_dtype = go.Figure(go.Pie(
            labels=dtype_df["dtype"], values=dtype_df["count"],
            marker=dict(colors=[C_PURPLE, C_ORANGE, C_GREEN, C_BLUE, C_RED]),
            textinfo="label+percent", hole=0.45,
        ))
        fig_dtype.update_layout(**PLOTLY_LAYOUT, title="Column Data Types", height=300)
        st.plotly_chart(fig_dtype, use_container_width=True)

    # Missing values
    with col_b:
        if len(eda["missing_df"]) > 0:
            miss = eda["missing_df"].head(12).reset_index()
            fig_miss = go.Figure(go.Bar(
                x=miss["Missing %"], y=miss["index"],
                orientation="h",
                marker_color=C_RED, opacity=0.85,
            ))
            fig_miss.update_layout(**PLOTLY_LAYOUT, title="Top Missing Value Columns (%)", height=300)
            fig_miss.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("No missing values detected in the dataset.")

    # Top events
    if eda["top_events"] is not None:
        top_ev = eda["top_events"].reset_index()
        top_ev.columns = ["event", "count"]
        fig_ev = go.Figure(go.Bar(
            x=top_ev["count"], y=top_ev["event"],
            orientation="h",
            marker=dict(color=top_ev["count"], colorscale=[[0, C_PURPLE], [1, C_BLUE]]),
            text=top_ev["count"].apply(lambda v: f"{v:,}"),
            textposition="outside",
        ))
        fig_ev.update_layout(**PLOTLY_LAYOUT, title="Top 15 Events by Frequency", height=380)
        fig_ev.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_ev, use_container_width=True)

    # Activity over time
    if eda["activity_over_time"] is not None:
        aot = eda["activity_over_time"]
        fig_aot = go.Figure(go.Scatter(
            x=aot["week"], y=aot["event_count"],
            mode="lines", fill="tozeroy",
            line=dict(color=C_PURPLE, width=2),
            fillcolor="rgba(167,139,250,0.15)",
        ))
        fig_aot.update_layout(**PLOTLY_LAYOUT, title="Weekly Event Activity Over Time", height=300)
        st.plotly_chart(fig_aot, use_container_width=True)

    # Browser / OS / Country
    if any(x is not None for x in [eda["browser_dist"], eda["os_dist"], eda["country_dist"]]):
        st.markdown("#### Device & Geography")
        ca, cb, cc = st.columns(3, gap="small")
        if eda["browser_dist"] is not None:
            with ca:
                bd = eda["browser_dist"].reset_index()
                bd.columns = ["browser", "count"]
                fig_br = go.Figure(go.Bar(
                    x=bd["count"], y=bd["browser"], orientation="h",
                    marker_color=C_BLUE, opacity=0.85,
                ))
                fig_br.update_layout(**PLOTLY_LAYOUT, title="Browser Distribution", height=320)
                fig_br.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_br, use_container_width=True)
        if eda["os_dist"] is not None:
            with cb:
                od = eda["os_dist"].reset_index()
                od.columns = ["os", "count"]
                fig_os = go.Figure(go.Bar(
                    x=od["count"], y=od["os"], orientation="h",
                    marker_color=C_GREEN, opacity=0.85,
                ))
                fig_os.update_layout(**PLOTLY_LAYOUT, title="OS Distribution", height=320)
                fig_os.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_os, use_container_width=True)
        if eda["country_dist"] is not None:
            with cc:
                ctd = eda["country_dist"].reset_index()
                ctd.columns = ["country", "count"]
                fig_ct = go.Figure(go.Bar(
                    x=ctd["count"], y=ctd["country"], orientation="h",
                    marker_color=C_ORANGE, opacity=0.85,
                ))
                fig_ct.update_layout(**PLOTLY_LAYOUT, title="Top Countries", height=320)
                fig_ct.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_ct, use_container_width=True)

    st.divider()

    # ── Section 2: User metrics ───────────────────────────────────────────────
    st.markdown("### 2 · User-Level Behavioral Metrics")

    col_c, col_d = st.columns(2, gap="medium")

    with col_c:
        fig_ev_dist = go.Figure()
        fig_ev_dist.add_trace(go.Histogram(
            x=user_base["total_events"].clip(upper=user_base["total_events"].quantile(0.99)),
            nbinsx=40, marker_color=C_PURPLE, opacity=0.8,
            name="Total Events",
        ))
        fig_ev_dist.update_layout(**PLOTLY_LAYOUT, title="Distribution of Lifetime Events per User",
                                  xaxis_title="Events", yaxis_title="Users", height=300)
        st.plotly_chart(fig_ev_dist, use_container_width=True)

    with col_d:
        fig_ten = go.Figure()
        fig_ten.add_trace(go.Histogram(
            x=user_base["tenure_days"].clip(upper=user_base["tenure_days"].quantile(0.99)),
            nbinsx=40, marker_color=C_BLUE, opacity=0.8,
            name="Tenure Days",
        ))
        fig_ten.update_layout(**PLOTLY_LAYOUT, title="Distribution of User Tenure (days)",
                              xaxis_title="Tenure (days)", yaxis_title="Users", height=300)
        st.plotly_chart(fig_ten, use_container_width=True)

    col_e, col_f = st.columns(2, gap="medium")

    with col_e:
        fig_dsl = go.Figure()
        fig_dsl.add_trace(go.Histogram(
            x=user_base["days_since_last"].clip(upper=user_base["days_since_last"].quantile(0.99)),
            nbinsx=40, marker_color=C_ORANGE, opacity=0.8,
        ))
        fig_dsl.update_layout(**PLOTLY_LAYOUT, title="Days Since Last Activity",
                              xaxis_title="Days", yaxis_title="Users", height=300)
        st.plotly_chart(fig_dsl, use_container_width=True)

    with col_f:
        fig_scatter = go.Figure(go.Scatter(
            x=user_base["tenure_days"].clip(upper=500),
            y=user_base["total_events"].clip(upper=user_base["total_events"].quantile(0.99)),
            mode="markers",
            marker=dict(
                color=user_base["days_since_last"].clip(upper=90),
                colorscale=[[0, C_GREEN], [0.5, C_ORANGE], [1, C_RED]],
                size=4, opacity=0.6,
                colorbar=dict(title=dict(text="Days Inactive", font=dict(color=TXT)), tickfont=dict(color=TXT)),
            ),
            text=user_base["user_id"] if "user_id" in user_base.columns else None,
        ))
        fig_scatter.update_layout(**PLOTLY_LAYOUT, title="Tenure vs. Total Events (coloured by inactivity)",
                                  xaxis_title="Tenure (days)", yaxis_title="Total Events", height=300)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── Section 3: Success labelling ─────────────────────────────────────────
    st.markdown("### 3 · Success Score & Labelling")

    m_ret, m_paid, m_dep, m_collab, m_power = st.columns(5)
    m_ret.metric("Long-term Retained",    f"{user_base['long_term_retention'].sum():,}",
                 delta=f"{user_base['long_term_retention'].mean()*100:.1f}%")
    m_paid.metric("Paid Users",           f"{user_base['is_paid_user'].sum():,}",
                  delta=f"{user_base['is_paid_user'].mean()*100:.1f}%")
    m_dep.metric("Deployed",              f"{user_base['has_deployment'].sum():,}",
                 delta=f"{user_base['has_deployment'].mean()*100:.1f}%")
    m_collab.metric("Collaborators (3+)", f"{user_base['collaboration_success'].sum():,}",
                    delta=f"{user_base['collaboration_success'].mean()*100:.1f}%")
    m_power.metric("Power Users",         f"{user_base['is_power_user'].sum():,}",
                   delta=f"{user_base['is_power_user'].mean()*100:.1f}%")

    col_g, col_h = st.columns(2, gap="medium")

    with col_g:
        lc = user_base["success_label"].value_counts().reindex(["Failed", "Moderate", "Successful"], fill_value=0)
        fig_lc = go.Figure(go.Bar(
            x=lc.index, y=lc.values,
            marker_color=[C_RED, C_ORANGE, C_GREEN],
            text=[f"{v:,}<br>({v/len(user_base)*100:.1f}%)" for v in lc.values],
            textposition="outside",
        ))
        fig_lc.update_layout(**PLOTLY_LAYOUT, title="Original Success Labels<br>(Strict Business Metrics)",
                             xaxis_title="Label", yaxis_title="Users", height=340)
        st.plotly_chart(fig_lc, use_container_width=True)

    with col_h:
        al = user_base["alternative_label"].value_counts().reindex(["Early/Churned", "Growing", "High Value"], fill_value=0)
        fig_al = go.Figure(go.Bar(
            x=al.index, y=al.values,
            marker_color=[C_RED, C_ORANGE, C_GREEN],
            text=[f"{v:,}<br>({v/len(user_base)*100:.1f}%)" for v in al.values],
            textposition="outside",
        ))
        fig_al.update_layout(**PLOTLY_LAYOUT, title="Alternative Labels<br>(Early-Stage Adjusted)",
                             xaxis_title="Label", yaxis_title="Users", height=340)
        st.plotly_chart(fig_al, use_container_width=True)

    # Success score histogram
    fig_score = go.Figure()
    fig_score.add_trace(go.Histogram(
        x=user_base["success_score"], nbinsx=30,
        marker_color=C_ORANGE, opacity=0.85, name="Success Score",
    ))
    fig_score.add_vline(x=30, line_color=C_GREEN,  line_dash="dash", line_width=2,
                        annotation_text="Moderate threshold", annotation_font_color=C_GREEN)
    fig_score.add_vline(x=60, line_color=C_RED,    line_dash="dash", line_width=2,
                        annotation_text="Successful threshold", annotation_font_color=C_RED)
    fig_score.update_layout(**PLOTLY_LAYOUT, title="Composite Success Score Distribution (0–100)",
                            xaxis_title="Score", yaxis_title="Users", height=300)
    st.plotly_chart(fig_score, use_container_width=True)

    # Alternative score histogram
    fig_alt_score = go.Figure()
    fig_alt_score.add_trace(go.Histogram(
        x=user_base["alternative_score"], nbinsx=30,
        marker_color=C_PURPLE, opacity=0.85, name="Alternative Score",
    ))
    fig_alt_score.add_vline(x=30, line_color=C_ORANGE, line_dash="dash", line_width=2,
                            annotation_text="Growing threshold", annotation_font_color=C_ORANGE)
    fig_alt_score.add_vline(x=60, line_color=C_GREEN,  line_dash="dash", line_width=2,
                            annotation_text="High Value threshold", annotation_font_color=C_GREEN)
    fig_alt_score.update_layout(**PLOTLY_LAYOUT, title="Alternative Success Score Distribution (Early-Stage Adjusted)",
                                xaxis_title="Score", yaxis_title="Users", height=300)
    st.plotly_chart(fig_alt_score, use_container_width=True)

    # Metric heatmap: avg characteristics by label
    chars = user_base.groupby("alternative_label")[
        ["total_events", "tenure_days", "days_since_last", "days_since_first"]
    ].mean().reindex(["Early/Churned", "Growing", "High Value"])

    chars_norm = (chars - chars.min()) / (chars.max() - chars.min() + 1e-9)
    fig_heat = go.Figure(go.Heatmap(
        z=chars_norm.values,
        x=["Total Events", "Tenure (days)", "Days Inactive", "Days Since First"],
        y=chars.index.tolist(),
        colorscale=[[0, "#1a1a22"], [0.5, C_PURPLE], [1, C_GREEN]],
        text=chars.round(1).values,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title=dict(text="Normalised", font=dict(color=TXT)), tickfont=dict(color=TXT)),
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, title="Avg User Characteristics by Label (normalised)",
                           height=280)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Section 4: Survival / Churn Analysis ─────────────────────────────────
    st.markdown("### 4 · Survival & Churn Analysis")

    churn_rate   = survival["churned"].mean() * 100
    active_users = int((survival["churned"] == 0).sum())
    churned_n    = int(survival["churned"].sum())

    ms1, ms2, ms3, ms4 = st.columns(4)
    ms1.metric("Overall churn rate",     f"{churn_rate:.1f}%")
    ms2.metric("Churned users",          f"{churned_n:,}")
    ms3.metric("Still active (censored)", f"{active_users:,}")
    ms4.metric("Median tenure (days)",   f"{survival['time_to_event'].median():.0f}")

    # Risk segment distribution
    col_i, col_j = st.columns(2, gap="medium")

    with col_i:
        rs = survival["risk_segment"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
        fig_rs = go.Figure(go.Bar(
            x=rs.index, y=rs.values,
            marker_color=[C_GREEN, C_ORANGE, C_RED],
            text=[f"{v:,}<br>({v/len(survival)*100:.1f}%)" for v in rs.values],
            textposition="outside",
        ))
        fig_rs.update_layout(**PLOTLY_LAYOUT, title="Risk Segment Distribution",
                             xaxis_title="Segment", yaxis_title="Users", height=320)
        st.plotly_chart(fig_rs, use_container_width=True)

    with col_j:
        el_churn = survival.groupby("engagement_level")["churned"].agg(["mean", "count"]).reset_index()
        el_churn.columns = ["level", "churn_rate", "count"]
        el_churn = el_churn.set_index("level").reindex(["Low", "Medium", "High"]).reset_index()
        el_churn["churn_rate"] *= 100
        fig_el = go.Figure()
        fig_el.add_trace(go.Bar(
            x=el_churn["level"], y=el_churn["churn_rate"],
            marker_color=[C_RED, C_ORANGE, C_GREEN],
            text=[f"{v:.1f}%" for v in el_churn["churn_rate"]], textposition="outside",
            name="Churn rate",
        ))
        fig_el.update_layout(**PLOTLY_LAYOUT, title="Churn Rate by Engagement Level",
                             xaxis_title="Engagement", yaxis_title="Churn Rate (%)", height=320)
        st.plotly_chart(fig_el, use_container_width=True)

    # Kaplan-Meier curves
    st.markdown("#### Kaplan-Meier Survival Curves")
    km_col1, km_col2, km_col3 = st.columns(3, gap="small")

    # KM by risk segment
    with km_col1:
        fig_km1 = go.Figure()
        for seg, col in zip(["Low Risk", "Medium Risk", "High Risk"], [C_GREEN, C_ORANGE, C_RED]):
            sd = survival[survival["risk_segment"] == seg]
            if len(sd) > 0:
                t, s = km_estimate(sd["time_to_event"].values, sd["churned"].values)
                fig_km1.add_trace(go.Scatter(
                    x=np.concatenate([[0], t]), y=np.concatenate([[1], s]),
                    mode="lines", name=seg, line=dict(color=col, width=2.5, shape="hv"),
                ))
        fig_km1.update_layout(**PLOTLY_LAYOUT, title="KM by Risk Segment",
                              xaxis_title="Days", yaxis_title="Survival Prob.", height=360)
        fig_km1.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_km1, use_container_width=True)

    # KM by engagement
    with km_col2:
        fig_km2 = go.Figure()
        for lvl, col in zip(["Low", "Medium", "High"], [C_RED, C_ORANGE, C_GREEN]):
            sd = survival[survival["engagement_level"] == lvl]
            if len(sd) > 0:
                t, s = km_estimate(sd["time_to_event"].values, sd["churned"].values)
                fig_km2.add_trace(go.Scatter(
                    x=np.concatenate([[0], t]), y=np.concatenate([[1], s]),
                    mode="lines", name=f"{lvl} Engagement",
                    line=dict(color=col, width=2.5, shape="hv"),
                ))
        fig_km2.update_layout(**PLOTLY_LAYOUT, title="KM by Engagement Level",
                              xaxis_title="Days", yaxis_title="Survival Prob.", height=360)
        fig_km2.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_km2, use_container_width=True)

    # KM by deployment
    with km_col3:
        fig_km3 = go.Figure()
        for stat, col in zip(["Has Deployment", "No Deployment"], [C_GREEN, C_ORANGE]):
            sd = survival[survival["deployment_status"] == stat]
            if len(sd) > 0:
                t, s = km_estimate(sd["time_to_event"].values, sd["churned"].values)
                fig_km3.add_trace(go.Scatter(
                    x=np.concatenate([[0], t]), y=np.concatenate([[1], s]),
                    mode="lines", name=stat,
                    line=dict(color=col, width=2.5, shape="hv"),
                ))
        fig_km3.update_layout(**PLOTLY_LAYOUT, title="KM by Deployment Status",
                              xaxis_title="Days", yaxis_title="Survival Prob.", height=360)
        fig_km3.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_km3, use_container_width=True)

    # Median survival table
    with st.expander("Median survival times by segment", expanded=False):
        rows = []
        for seg in ["Low Risk", "Medium Risk", "High Risk"]:
            sd = survival[survival["risk_segment"] == seg]
            if len(sd) > 0:
                t, s = km_estimate(sd["time_to_event"].values, sd["churned"].values)
                idx = np.where(s <= 0.5)[0]
                med = f"{t[idx[0]]:.0f} days" if len(idx) else "Not reached"
                rows.append({"Segment": seg, "N": len(sd),
                             "Churn Rate": f"{sd['churned'].mean()*100:.1f}%",
                             "Median Survival": med})
        for lvl in ["Low", "Medium", "High"]:
            sd = survival[survival["engagement_level"] == lvl]
            if len(sd) > 0:
                t, s = km_estimate(sd["time_to_event"].values, sd["churned"].values)
                idx = np.where(s <= 0.5)[0]
                med = f"{t[idx[0]]:.0f} days" if len(idx) else "Not reached"
                rows.append({"Segment": f"{lvl} Engagement", "N": len(sd),
                             "Churn Rate": f"{sd['churned'].mean()*100:.1f}%",
                             "Median Survival": med})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 5: Cohort analysis ────────────────────────────────────────────
    st.markdown("### 5 · Cohort & Tenure Analysis")

    # Tenure buckets
    user_base["tenure_bucket"] = pd.cut(
        user_base["tenure_days"],
        bins=[-1, 7, 30, 90, 180, 365, 9999],
        labels=["< 1 week", "1–4 weeks", "1–3 months", "3–6 months", "6–12 months", "12+ months"],
    )
    tb = user_base.groupby("tenure_bucket", observed=True).agg(
        users=("user_id", "count"),
        churn_rate=("churned" if "churned" in user_base.columns else "days_since_last",
                    lambda x: (x > 30).mean() * 100 if x.name == "days_since_last" else x.mean() * 100),
        avg_events=("total_events", "mean"),
    ).reset_index()

    # Recalculate churn by days_since_last
    user_base["is_churned"] = user_base["days_since_last"] > 30
    tb2 = user_base.groupby("tenure_bucket", observed=True).agg(
        users=("user_id", "count"),
        churn_rate=("is_churned", lambda x: x.mean() * 100),
        avg_events=("total_events", "mean"),
    ).reset_index()

    fig_cohort = make_subplots(specs=[[{"secondary_y": True}]])
    fig_cohort.add_trace(go.Bar(
        x=tb2["tenure_bucket"].astype(str), y=tb2["users"],
        name="Users", marker_color=C_PURPLE, opacity=0.7,
    ), secondary_y=False)
    fig_cohort.add_trace(go.Scatter(
        x=tb2["tenure_bucket"].astype(str), y=tb2["churn_rate"],
        mode="lines+markers", name="Churn Rate (%)",
        line=dict(color=C_RED, width=2.5),
        marker=dict(size=8),
    ), secondary_y=True)
    fig_cohort.update_layout(**PLOTLY_LAYOUT, title="User Count & Churn Rate by Tenure Bucket", height=360)
    fig_cohort.update_yaxes(title_text="Users", secondary_y=False, title_font=dict(color=C_PURPLE))
    fig_cohort.update_yaxes(title_text="Churn Rate (%)", secondary_y=True, title_font=dict(color=C_RED))
    st.plotly_chart(fig_cohort, use_container_width=True)

    col_k, col_l = st.columns(2, gap="medium")
    with col_k:
        fig_avg_ev = go.Figure(go.Bar(
            x=tb2["tenure_bucket"].astype(str), y=tb2["avg_events"].round(1),
            marker=dict(color=tb2["avg_events"],
                        colorscale=[[0, C_BLUE], [1, C_GREEN]]),
            text=tb2["avg_events"].round(0).astype(int),
            textposition="outside",
        ))
        fig_avg_ev.update_layout(**PLOTLY_LAYOUT, title="Avg Lifetime Events by Tenure Bucket",
                                 xaxis_title="Tenure", yaxis_title="Avg Events", height=320)
        st.plotly_chart(fig_avg_ev, use_container_width=True)

    with col_l:
        # Days since last vs events scatter by alternative label
        sample = user_base.sample(min(3000, len(user_base)), random_state=42)
        color_map = {"Early/Churned": C_RED, "Growing": C_ORANGE, "High Value": C_GREEN}
        fig_scatter2 = go.Figure()
        for lbl, col in color_map.items():
            sub = sample[sample["alternative_label"] == lbl]
            fig_scatter2.add_trace(go.Scatter(
                x=sub["days_since_last"].clip(upper=200),
                y=sub["total_events"].clip(upper=sub["total_events"].quantile(0.99)),
                mode="markers", name=lbl,
                marker=dict(color=col, size=4, opacity=0.6),
            ))
        fig_scatter2.update_layout(**PLOTLY_LAYOUT, title="Days Inactive vs Events (by User Label)",
                                   xaxis_title="Days Since Last Activity",
                                   yaxis_title="Total Events", height=320)
        st.plotly_chart(fig_scatter2, use_container_width=True)

    st.divider()

    # ── Section 6: Raw data explorer ─────────────────────────────────────────
    st.markdown("### 6 · User Profile Explorer")
    with st.expander("Browse user-level summary table", expanded=False):
        show_cols = ["user_id", "total_events", "tenure_days", "days_since_first",
                     "days_since_last", "success_score", "success_label",
                     "alternative_label", "long_term_retention", "is_power_user",
                     "is_paid_user", "has_deployment", "collaboration_success"]
        show_cols = [c for c in show_cols if c in user_base.columns]
        st.dataframe(
            user_base[show_cols]
            .sort_values("success_score", ascending=False)
            .reset_index(drop=True),
            use_container_width=True, height=400,
        )