"""Future-holdout success-label retrain with RICHER features (§7 item 15).

Extends ``scripts/future_holdout_retrain.py`` by layering session-pattern,
workflow n-gram, and engagement-momentum features on top of the 4 base
features. Everything is computed strictly from events inside the feature
window so the model cannot observe the target.

Same split / balancing / ensemble shape as the base script; writes a parallel
JSON summary to ``outputs/_future_holdout_89_9_rich.json`` so the two runs can
be compared apples-to-apples.

Usage (from repo root)::

    python scripts/future_holdout_retrain_rich.py \
        --json-out outputs/_future_holdout_89_9_rich.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "user_retention.parquet"

# ── Feature groups ──────────────────────────────────────────────────────────
BASE_FEATURES = [
    "total_events",
    "tenure_days",
    "days_since_first",
    "days_since_last",
]

SESSION_FEATURES = [
    "total_sessions",
    "avg_session_length_min",
    "max_session_length_min",
    "std_session_length_min",
    "avg_events_per_session",
    "avg_event_density",
    "deep_work_sessions",
    "deep_work_ratio",
    "avg_inter_session_gap_hours",
]

WORKFLOW_FEATURES = [
    "unique_event_types",
    "power_events_count",
    "has_deployment_sequence",
    "has_agent_workflow",
    "has_create_run_pattern",
    "error_count",
    "failed_block_runs",
    "repeated_events_count",
    "sequence_diversity",
    "trigram_count",
    "power_user_score",
    "struggle_score",
]

MOMENTUM_FEATURES = [
    "active_days",
    "last7d_events",
    "prior7d_events",
    "momentum_change",
    "accel_periods",
    "decel_periods",
    "max_daily_events",
    "events_stdev",
]

FEATURES = BASE_FEATURES + SESSION_FEATURES + WORKFLOW_FEATURES + MOMENTUM_FEATURES


POWER_EVENT_PATTERNS = {
    "block_create",
    "block_run",
    "block_execution_success",
    "agent_accept_suggestion",
    "agent_worker_created",
    "agent_worker_finished",
    "canvas_created",
    "canvas_shared",
    "api_deployed",
    "model_trained",
    "model_deployed",
}
STRUGGLE_EVENT_PATTERNS = {
    "block_execution_failed",
    "error_occurred",
    "exception_caught",
    "connection_failed",
    "query_failed",
}


# ─── Session-pattern features ──────────────────────────────────────────────

def _session_features(feat_events: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """Per-user session aggregates. Mirrors `Development/Session Pattern Analysis.py`
    but operates strictly on the feature-window event slice."""
    df = feat_events.sort_values(["distinct_id", "timestamp"]).copy()
    df["gap"] = df.groupby("distinct_id")["timestamp"].diff()
    df["is_new_session"] = (df["gap"] > pd.Timedelta(minutes=gap_minutes)) | df["gap"].isna()
    df["session_number"] = df.groupby("distinct_id")["is_new_session"].cumsum()
    df["session_id"] = df["distinct_id"] + "_s" + df["session_number"].astype(str)

    sess = (
        df.groupby(["distinct_id", "session_id"])
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            events_in_session=("timestamp", "size"),
        )
        .reset_index()
    )
    sess["session_length_min"] = (
        (sess["session_end"] - sess["session_start"]).dt.total_seconds() / 60.0
    )
    sess["event_density"] = sess["events_in_session"] / (sess["session_length_min"] + 1.0)

    # Deep-work flag: top-quartile length AND top-quartile density.
    dur_thr = sess["session_length_min"].quantile(0.75)
    den_thr = sess["event_density"].quantile(0.75)
    sess["is_deep_work"] = (sess["session_length_min"] >= dur_thr) & (
        sess["event_density"] >= den_thr
    )

    # Inter-session gaps (ordered by start time within user).
    sess = sess.sort_values(["distinct_id", "session_start"]).reset_index(drop=True)
    sess["next_start"] = sess.groupby("distinct_id")["session_start"].shift(-1)
    sess["inter_gap_h"] = (
        (sess["next_start"] - sess["session_end"]).dt.total_seconds() / 3600.0
    )

    agg = (
        sess.groupby("distinct_id")
        .agg(
            total_sessions=("session_id", "count"),
            avg_session_length_min=("session_length_min", "mean"),
            max_session_length_min=("session_length_min", "max"),
            std_session_length_min=("session_length_min", "std"),
            avg_events_per_session=("events_in_session", "mean"),
            avg_event_density=("event_density", "mean"),
            deep_work_sessions=("is_deep_work", "sum"),
            avg_inter_session_gap_hours=("inter_gap_h", "mean"),
        )
        .reset_index()
        .rename(columns={"distinct_id": "user_id"})
    )
    agg["std_session_length_min"] = agg["std_session_length_min"].fillna(0.0)
    agg["avg_inter_session_gap_hours"] = agg["avg_inter_session_gap_hours"].fillna(0.0)
    agg["deep_work_ratio"] = agg["deep_work_sessions"] / agg["total_sessions"].clip(lower=1)
    return agg


# ─── Workflow n-gram / power & struggle features ──────────────────────────

def _workflow_features(feat_events: pd.DataFrame) -> pd.DataFrame:
    """Per-user n-gram + power/struggle signals. Mirrors the scoring logic
    in `Development/Workflow Sequence Patterns.py`."""
    df = feat_events.sort_values(["distinct_id", "timestamp"])
    user_events = df.groupby("distinct_id")["event"].apply(list)

    rows = []
    for uid, events in user_events.items():
        n = len(events)
        event_counts = Counter(events)

        power_events_count = sum(
            event_counts.get(e, 0) for e in POWER_EVENT_PATTERNS
        )
        has_deployment = int(any("deploy" in str(e).lower() for e in event_counts))
        has_agent = int(any("agent" in str(e).lower() for e in event_counts))

        # has_create_run_pattern: adjacency of create→run or created→started.
        has_create_run = 0
        for i in range(n - 1):
            pair = (events[i], events[i + 1])
            if pair == ("block_create", "block_run") or pair == (
                "block_created",
                "block_execution_started",
            ):
                has_create_run = 1
                break

        error_count = sum(event_counts.get(e, 0) for e in STRUGGLE_EVENT_PATTERNS)
        failed_block_runs = event_counts.get("block_execution_failed", 0)

        # Repeated-event bursts: three identical events in a row.
        repeated_events = 0
        for i in range(n - 2):
            if events[i] == events[i + 1] == events[i + 2]:
                repeated_events += 1

        # Trigram diversity.
        trigrams = [
            (events[i], events[i + 1], events[i + 2]) for i in range(n - 2)
        ] if n >= 3 else []
        trigram_count = len(trigrams)
        sequence_diversity = (
            len(set(trigrams)) / trigram_count if trigram_count else 0.0
        )

        power_user_score = (
            power_events_count * 0.4
            + has_deployment * 10
            + has_agent * 5
            + has_create_run * 8
        )
        struggle_score = (
            error_count * 2 + failed_block_runs * 3 + repeated_events * 1.5
        )

        rows.append(
            {
                "user_id": uid,
                "unique_event_types": len(event_counts),
                "power_events_count": power_events_count,
                "has_deployment_sequence": has_deployment,
                "has_agent_workflow": has_agent,
                "has_create_run_pattern": has_create_run,
                "error_count": error_count,
                "failed_block_runs": failed_block_runs,
                "repeated_events_count": repeated_events,
                "sequence_diversity": sequence_diversity,
                "trigram_count": trigram_count,
                "power_user_score": power_user_score,
                "struggle_score": struggle_score,
            }
        )
    return pd.DataFrame(rows)


# ─── Engagement-momentum features ─────────────────────────────────────────

def _momentum_features(
    feat_events: pd.DataFrame,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    """7-day / prior-7-day / velocity counts referenced to the end of the
    feature window. Equivalent to the feature-window-restricted version of
    `Development/Engagement Momentum Tracking.py`."""
    df = feat_events.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()

    daily = (
        df.groupby(["distinct_id", "date"]).size().reset_index(name="daily_events")
    )

    ref_date = split_date - pd.Timedelta(days=1)  # last calendar day in window
    cutoff_last7 = ref_date - pd.Timedelta(days=6)
    cutoff_prior7_start = ref_date - pd.Timedelta(days=13)
    cutoff_prior7_end = ref_date - pd.Timedelta(days=7)

    last7 = daily[daily["date"] >= cutoff_last7]
    prior7 = daily[(daily["date"] >= cutoff_prior7_start) & (daily["date"] <= cutoff_prior7_end)]

    last7_agg = last7.groupby("distinct_id")["daily_events"].sum().rename("last7d_events")
    prior7_agg = prior7.groupby("distinct_id")["daily_events"].sum().rename("prior7d_events")

    # Active-day count, max single-day events, daily-events stdev.
    base_agg = daily.groupby("distinct_id").agg(
        active_days=("date", "nunique"),
        max_daily_events=("daily_events", "max"),
        events_stdev=("daily_events", "std"),
    )

    # Accel / decel periods: signs of the day-over-day delta on daily_events.
    # For each user, compute diffs along date order and count +/- transitions.
    accel = {}
    decel = {}
    for uid, sub in daily.sort_values(["distinct_id", "date"]).groupby("distinct_id"):
        vals = sub["daily_events"].values
        if len(vals) < 2:
            accel[uid] = 0
            decel[uid] = 0
            continue
        diffs = np.diff(vals)
        accel[uid] = int((diffs > 0).sum())
        decel[uid] = int((diffs < 0).sum())
    accel_s = pd.Series(accel, name="accel_periods")
    decel_s = pd.Series(decel, name="decel_periods")

    mom = (
        base_agg.join(last7_agg, how="left")
        .join(prior7_agg, how="left")
        .join(accel_s, how="left")
        .join(decel_s, how="left")
        .reset_index()
        .rename(columns={"distinct_id": "user_id"})
    )
    for col in ["last7d_events", "prior7d_events", "accel_periods", "decel_periods"]:
        mom[col] = mom[col].fillna(0).astype(float)
    mom["events_stdev"] = mom["events_stdev"].fillna(0.0)
    mom["momentum_change"] = mom["last7d_events"] - mom["prior7d_events"]
    return mom


# ─── Feature/label frame assembly ─────────────────────────────────────────

def build_rich_frame(
    events: pd.DataFrame, feature_horizon_days: int
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    dataset_start = events["timestamp"].min().normalize()
    dataset_end = events["timestamp"].max().normalize()
    total_days = (dataset_end - dataset_start).days + 1
    if feature_horizon_days >= total_days:
        raise ValueError(
            f"feature_horizon_days={feature_horizon_days} must be < total dataset span ({total_days} days)"
        )

    split_date = dataset_start + pd.Timedelta(days=feature_horizon_days)
    feat_events = events[events["timestamp"] < split_date]
    target_events = events[events["timestamp"] >= split_date]

    # ── Base features (identical to scripts/future_holdout_retrain.py) ──
    feat_agg = (
        feat_events.groupby("distinct_id")
        .agg(
            total_events=("event", "size"),
            first_event=("timestamp", "min"),
            last_event=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"distinct_id": "user_id"})
    )
    ref_ts = feat_events["timestamp"].max()
    feat_agg["tenure_days"] = (feat_agg["last_event"] - feat_agg["first_event"]).dt.days.astype(float)
    feat_agg["days_since_first"] = (ref_ts - feat_agg["first_event"]).dt.days.astype(float)
    feat_agg["days_since_last"] = (ref_ts - feat_agg["last_event"]).dt.days.astype(float)
    base = feat_agg[["user_id", *BASE_FEATURES]].copy()

    # ── Rich feature groups ──
    sess = _session_features(feat_events)
    wf = _workflow_features(feat_events)
    mom = _momentum_features(feat_events, split_date)

    frame = (
        base.merge(sess, on="user_id", how="left")
        .merge(wf, on="user_id", how="left")
        .merge(mom, on="user_id", how="left")
    )

    # ── Target ──
    target_users = set(target_events["distinct_id"].unique())
    frame["target"] = frame["user_id"].isin(target_users).astype(int)

    # Fill any residual NaNs (users with a single event -> no session stats etc.)
    for col in FEATURES:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(float)

    frame = frame[["user_id", *FEATURES, "target"]].copy()
    return frame, dataset_start, split_date, dataset_end


# ─── Split / balance / train (mirror of base script) ───────────────────────

def split_and_balance(frame: pd.DataFrame, seed: int = 42):
    X = frame[FEATURES].values.astype(float)
    y = frame["target"].values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    df_tr = pd.DataFrame(X_train_s, columns=FEATURES)
    df_tr["target"] = y_train
    maj = df_tr[df_tr["target"] == 0]
    minr = df_tr[df_tr["target"] == 1]
    if len(minr) == 0 or len(maj) == 0:
        raise RuntimeError("Training split has a single-class distribution; cannot balance.")
    if len(minr) < len(maj):
        minr_up = resample(minr, replace=True, n_samples=len(maj), random_state=seed)
        bal = pd.concat([maj, minr_up])
    else:
        maj_up = resample(maj, replace=True, n_samples=len(minr), random_state=seed)
        bal = pd.concat([maj_up, minr])
    bal = bal.sample(frac=1, random_state=seed).reset_index(drop=True)
    X_tr_bal = bal[FEATURES].values
    y_tr_bal = bal["target"].values

    n_samples = len(y_train)
    class_weight = {
        0: n_samples / (2 * max((y_train == 0).sum(), 1)),
        1: n_samples / (2 * max((y_train == 1).sum(), 1)),
    }
    return {
        "X_tr_bal": X_tr_bal,
        "y_tr_bal": y_tr_bal,
        "X_val": X_val_s,
        "y_val": y_val,
        "X_test": X_test_s,
        "y_test": y_test,
        "class_weight": class_weight,
    }


def train_ensemble(data: dict, seed: int = 42):
    cw = data["class_weight"]
    Xtr, ytr = data["X_tr_bal"], data["y_tr_bal"]
    Xv, yv = data["X_val"], data["y_val"]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight=cw, random_state=seed, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        learning_rate=0.05, n_estimators=300, max_depth=5, random_state=seed
    )
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=200,
        learning_rate=0.05,
        random_state=seed,
    )
    lr = LogisticRegression(class_weight=cw, max_iter=1000, random_state=seed)

    base_models = {"rf": rf, "gb": gb, "ada": ada, "lr": lr}
    val_f1s: dict[str, float] = {}
    val_aucs: dict[str, float] = {}
    for name, m in base_models.items():
        m.fit(Xtr, ytr)
        val_f1s[name] = f1_score(yv, m.predict(Xv), average="weighted")
        val_aucs[name] = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])

    weights = [val_f1s[n] for n in ["rf", "gb", "ada", "lr"]]
    voting = VotingClassifier(
        estimators=[(n, base_models[n]) for n in ["rf", "gb", "ada", "lr"]],
        voting="soft",
        weights=[w / sum(weights) for w in weights],
    )
    voting.fit(Xtr, ytr)
    val_f1s["voting"] = f1_score(yv, voting.predict(Xv), average="weighted")
    val_aucs["voting"] = roc_auc_score(yv, voting.predict_proba(Xv)[:, 1])

    stacking = StackingClassifier(
        estimators=[(n, base_models[n]) for n in ["rf", "gb", "ada", "lr"]],
        final_estimator=LogisticRegression(class_weight=cw, max_iter=1000, random_state=seed),
        cv=5,
    )
    stacking.fit(Xtr, ytr)
    val_f1s["stacking"] = f1_score(yv, stacking.predict(Xv), average="weighted")
    val_aucs["stacking"] = roc_auc_score(yv, stacking.predict_proba(Xv)[:, 1])

    all_models = {**base_models, "voting": voting, "stacking": stacking}
    best_by_f1 = max(val_f1s, key=val_f1s.get)
    best_by_auc = max(val_aucs, key=val_aucs.get)
    return all_models, val_f1s, val_aucs, best_by_f1, best_by_auc


def evaluate_on_test(model, data: dict) -> dict:
    Xt, yt = data["X_test"], data["y_test"]
    yp = model.predict(Xt)
    yproba = model.predict_proba(Xt)[:, 1]
    return {
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, average="weighted", zero_division=0),
        "recall": recall_score(yt, yp, average="weighted", zero_division=0),
        "f1": f1_score(yt, yp, average="weighted"),
        "roc_auc": roc_auc_score(yt, yproba),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        "n_test": int(len(yt)),
        "test_positive_rate": float((yt == 1).mean()),
    }


def feature_importance_from_best(model, feature_names: list[str]) -> list[dict]:
    """Return top-N features by importance/abs coefficient, if exposed."""
    imps = None
    if hasattr(model, "feature_importances_"):
        imps = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        imps = np.abs(np.asarray(model.coef_)).ravel()
    if imps is None or len(imps) != len(feature_names):
        return []
    order = np.argsort(-imps)
    return [
        {"feature": feature_names[i], "importance": float(imps[i])}
        for i in order
    ]


# ─── Driver ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    ap.add_argument("--feature-horizon-days", type=int, default=89)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--json-out",
        default=str(REPO_ROOT / "outputs" / "_future_holdout_89_9_rich.json"),
    )
    args = ap.parse_args()

    print("=" * 78)
    print("FUTURE-HOLDOUT RICH-FEATURES RETRAIN (pipeline_deep_dive.md §7 item 15)")
    print("=" * 78)

    events = pd.read_parquet(args.parquet)
    print(f"\nLoaded {len(events):,} events from {args.parquet}")

    t0 = time.perf_counter()
    frame, dataset_start, split_date, dataset_end = build_rich_frame(
        events, feature_horizon_days=args.feature_horizon_days
    )
    total_days = (dataset_end - dataset_start).days + 1
    target_days = total_days - args.feature_horizon_days
    print(
        f"\nDataset span : {dataset_start.date()} → {dataset_end.date()} ({total_days} days)"
    )
    print(
        f"Feature window: {dataset_start.date()} → {(split_date - pd.Timedelta(days=1)).date()} ({args.feature_horizon_days} days)"
    )
    print(f"Target window : {split_date.date()} → {dataset_end.date()} ({target_days} days)")

    n_users = len(frame)
    pos_rate = frame["target"].mean()
    print(f"\nUsers with activity in feature window: {n_users:,}")
    print(f"Active in target window (label=1)     : {int(frame['target'].sum()):,} ({pos_rate * 100:.2f}%)")
    print(f"Inactive in target window (label=0)   : {int((frame['target'] == 0).sum()):,}")
    print(f"Feature columns ({len(FEATURES)}): {FEATURES}")
    print(f"✓ Built rich frame in {time.perf_counter() - t0:.1f}s")

    if pos_rate == 0 or pos_rate == 1:
        print("\nDegenerate label distribution; aborting.")
        return 1

    t1 = time.perf_counter()
    data = split_and_balance(frame, seed=args.seed)
    all_models, val_f1s, val_aucs, best_by_f1, best_by_auc = train_ensemble(data, seed=args.seed)
    print(f"\n✓ Trained 6 models in {time.perf_counter() - t1:.1f}s")

    print("\nValidation metrics (weighted F1 | ROC-AUC):")
    for name in sorted(val_f1s, key=lambda k: -val_aucs[k]):
        print(f"   {name:12s}  F1={val_f1s[name]:.4f}  AUC={val_aucs[name]:.4f}")
    print(f"\n🏆 Best by val F1 : {best_by_f1}   (F1={val_f1s[best_by_f1]:.4f})")
    print(f"🏆 Best by val AUC: {best_by_auc}  (AUC={val_aucs[best_by_auc]:.4f})")

    # Evaluate every model on the test set so we can honestly compare headlines.
    test_by_model: dict[str, dict] = {}
    print("\nTest metrics per model:")
    print(f"   {'model':<12s} {'acc':>7s} {'f1':>7s} {'AUC':>7s}")
    for name, mdl in all_models.items():
        tm = evaluate_on_test(mdl, data)
        test_by_model[name] = tm
        print(f"   {name:<12s} {tm['accuracy']:.4f} {tm['f1']:.4f} {tm['roc_auc']:.4f}")

    best_model = all_models[best_by_auc]
    test_metrics = test_by_model[best_by_auc]

    # Baselines for comparison
    baseline_composite_acc = 0.9951
    baseline_composite_auc = 0.9952
    holdout_base_acc = 0.9044
    holdout_base_auc = 0.9414

    delta_vs_composite_auc = test_metrics["roc_auc"] - baseline_composite_auc
    delta_vs_holdout_base_auc = test_metrics["roc_auc"] - holdout_base_auc

    print("\nComparison (best model by val ROC-AUC):")
    print(f"   composite-label baseline   ROC-AUC : {baseline_composite_auc:.4f}")
    print(f"   base 4-feature holdout     ROC-AUC : {holdout_base_auc:.4f}")
    print(f"   RICH {len(FEATURES)}-feature holdout   ROC-AUC : {test_metrics['roc_auc']:.4f}")
    print(f"   Δ vs. composite baseline          : {delta_vs_composite_auc:+.4f}")
    print(f"   Δ vs. base 4-feature holdout      : {delta_vs_holdout_base_auc:+.4f}  ← lift from richer features")

    # Also report the best-test-AUC model (even if not selected), for transparency.
    best_test_auc_name = max(test_by_model, key=lambda k: test_by_model[k]["roc_auc"])
    best_test_auc = test_by_model[best_test_auc_name]["roc_auc"]
    print(
        f"\n(Informational) Best test ROC-AUC over all trained models: {best_test_auc_name} = {best_test_auc:.4f}"
    )

    importances = feature_importance_from_best(best_model, FEATURES)
    if importances:
        print("\nTop 10 features by importance (best-by-val-AUC model):")
        for rec in importances[:10]:
            print(f"   {rec['feature']:32s}  {rec['importance']:.4f}")

    summary = {
        "feature_horizon_days": args.feature_horizon_days,
        "dataset_start": str(dataset_start.date()),
        "split_date": str(split_date.date()),
        "dataset_end": str(dataset_end.date()),
        "n_users": int(n_users),
        "target_positive_rate": float(pos_rate),
        "best_model_by_val_auc": best_by_auc,
        "best_model_by_val_f1": best_by_f1,
        "best_model_by_test_auc": best_test_auc_name,
        "val_f1_per_model": {k: float(v) for k, v in val_f1s.items()},
        "val_auc_per_model": {k: float(v) for k, v in val_aucs.items()},
        "test_metrics": {
            k: (v if isinstance(v, (int, float, list)) else float(v))
            for k, v in test_metrics.items()
        },
        "test_metrics_per_model": {
            mname: {
                k: (v if isinstance(v, (int, float, list)) else float(v))
                for k, v in tm.items()
            }
            for mname, tm in test_by_model.items()
        },
        "comparison": {
            "composite_label_baseline_roc_auc": baseline_composite_auc,
            "composite_label_baseline_accuracy": baseline_composite_acc,
            "base_holdout_roc_auc": holdout_base_auc,
            "base_holdout_accuracy": holdout_base_acc,
            "delta_roc_auc_vs_composite_baseline": float(delta_vs_composite_auc),
            "delta_roc_auc_vs_base_holdout": float(delta_vs_holdout_base_auc),
        },
        "features_used": FEATURES,
        "feature_groups": {
            "base": BASE_FEATURES,
            "session": SESSION_FEATURES,
            "workflow": WORKFLOW_FEATURES,
            "momentum": MOMENTUM_FEATURES,
        },
        "top_feature_importances": importances[:20],
        "label_definition": "user has >=1 event in the target window",
        "notes": (
            "All features computed strictly on feature-window events "
            "(days 1..feature_horizon_days); target derived from the disjoint "
            "future window. No label-time information leaks into features."
        ),
    }

    print("\nJSON summary (omitting verbose per-model / importance fields):")
    verbose_keys = {"top_feature_importances", "test_metrics_per_model"}
    print(json.dumps({k: v for k, v in summary.items() if k not in verbose_keys}, indent=2))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n✓ Wrote JSON summary to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
