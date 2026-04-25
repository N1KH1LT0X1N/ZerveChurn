"""Future-holdout success-label retrain (pipeline_deep_dive.md §3.4).

Builds a leakage-free successor to the existing 99.51% ensemble by using a
*future* slice of the event log as the target instead of a score that was
derived from the same continuous features the model receives.

Setup:

- Feature window = days 1..FEATURE_HORIZON_DAYS of the event log.
- Target window  = days FEATURE_HORIZON_DAYS+1 .. end of the event log.
- Target         = `1` if the user has >=1 event in the target window.
- Features are computed strictly from events inside the feature window
  (total_events, tenure_days, days_since_first, days_since_last), so the
  model cannot observe the target.

Trains the same {RF, GB, Ada, LR, Voting, Stacking} ensemble the main
pipeline uses (`Development/02_base_models_ensemble.py`) for an apples-to-
apples comparison against the current composite-label number.

Usage (from repo root):

    python scripts/future_holdout_retrain.py

Writes a structured summary line to stdout. Does NOT overwrite
`ensemble_models.pkl`.
"""
from __future__ import annotations

import argparse
import json
import time
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

FEATURES = ["total_events", "tenure_days", "days_since_first", "days_since_last"]


# ─── Data preparation ───────────────────────────────────────────────────────

def build_feature_label_frame(
    events: pd.DataFrame,
    feature_horizon_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Split events into feature/target windows, compute per-user features+label.

    Returns `(frame, dataset_start, split_date, dataset_end)`. `frame` has
    columns `['user_id', *FEATURES, 'target']`. Users with zero events in the
    feature window are excluded — the model can only make a prediction if it
    has observed behaviour.
    """
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    dataset_start = events["timestamp"].min().normalize()
    dataset_end = events["timestamp"].max().normalize()
    total_days = (dataset_end - dataset_start).days + 1

    if feature_horizon_days >= total_days:
        raise ValueError(
            f"feature_horizon_days={feature_horizon_days} must be < total "
            f"dataset span ({total_days} days)"
        )

    # split_date is the first instant that belongs to the target window
    split_date = dataset_start + pd.Timedelta(days=feature_horizon_days)

    feat_events = events[events["timestamp"] < split_date]
    target_events = events[events["timestamp"] >= split_date]

    # Features from the feature-window events only
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

    # Reference = feature-window max timestamp (matches the wall-clock-fix
    # pattern already used by `Primary Success Metrics.py:19` and friends).
    ref_ts = feat_events["timestamp"].max()

    feat_agg["tenure_days"] = (feat_agg["last_event"] - feat_agg["first_event"]).dt.days.astype(float)
    feat_agg["days_since_first"] = (ref_ts - feat_agg["first_event"]).dt.days.astype(float)
    feat_agg["days_since_last"] = (ref_ts - feat_agg["last_event"]).dt.days.astype(float)

    # Target = user active in the held-out window?
    target_users = set(target_events["distinct_id"].unique())
    feat_agg["target"] = feat_agg["user_id"].isin(target_users).astype(int)

    frame = feat_agg[["user_id", *FEATURES, "target"]].copy()
    return frame, dataset_start, split_date, dataset_end


# ─── Ensemble training (mirrors Development/02_base_models_ensemble.py) ────

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

    # Upsample minority in training only
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
        "scaler": scaler,
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
    lr = LogisticRegression(class_weight=cw, max_iter=1000, random_state=seed, n_jobs=-1)

    base_models = {"rf": rf, "gb": gb, "ada": ada, "lr": lr}
    val_f1s: dict[str, float] = {}
    for name, m in base_models.items():
        m.fit(Xtr, ytr)
        val_f1s[name] = f1_score(yv, m.predict(Xv), average="weighted")

    weights = [val_f1s[n] for n in ["rf", "gb", "ada", "lr"]]
    voting = VotingClassifier(
        estimators=[(n, base_models[n]) for n in ["rf", "gb", "ada", "lr"]],
        voting="soft",
        weights=[w / sum(weights) for w in weights],
    )
    voting.fit(Xtr, ytr)
    val_f1s["voting"] = f1_score(yv, voting.predict(Xv), average="weighted")

    stacking = StackingClassifier(
        estimators=[(n, base_models[n]) for n in ["rf", "gb", "ada", "lr"]],
        final_estimator=LogisticRegression(
            class_weight=cw, max_iter=1000, random_state=seed
        ),
        cv=5,
    )
    stacking.fit(Xtr, ytr)
    val_f1s["stacking"] = f1_score(yv, stacking.predict(Xv), average="weighted")

    all_models = {**base_models, "voting": voting, "stacking": stacking}
    best_name = max(val_f1s, key=val_f1s.get)
    return all_models, val_f1s, best_name


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


# ─── Driver ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--parquet",
        default=str(DEFAULT_PARQUET),
        help="Path to user_retention.parquet",
    )
    ap.add_argument(
        "--feature-horizon-days",
        type=int,
        default=89,
        help="Number of days from the start to use as the feature window (target = remainder). Default: 89 (i.e. features Sep 1 - Nov 28, target Nov 29 - Dec 8 on the current 98-day dataset).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--json-out",
        default=None,
        help="Optional path to dump the full results JSON",
    )
    args = ap.parse_args()

    print("=" * 78)
    print("FUTURE-HOLDOUT SUCCESS-LABEL RETRAIN (pipeline_deep_dive.md §3.4)")
    print("=" * 78)

    events = pd.read_parquet(args.parquet)
    print(f"\nLoaded {len(events):,} events from {args.parquet}")

    frame, dataset_start, split_date, dataset_end = build_feature_label_frame(
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
    print(
        f"Target window : {split_date.date()} → {dataset_end.date()} ({target_days} days)"
    )

    n_users = len(frame)
    pos_rate = frame["target"].mean()
    print(f"\nUsers with activity in feature window: {n_users:,}")
    print(f"Active in target window (label=1)     : {frame['target'].sum():,} ({pos_rate * 100:.2f}%)")
    print(f"Inactive in target window (label=0)   : {(frame['target'] == 0).sum():,}")

    if pos_rate == 0 or pos_rate == 1:
        print("\nDegenerate label distribution; aborting.")
        return 1

    t0 = time.perf_counter()
    data = split_and_balance(frame, seed=args.seed)
    all_models, val_f1s, best_name = train_ensemble(data, seed=args.seed)
    print(f"\n✓ Trained 6 models in {time.perf_counter() - t0:.1f}s")

    print("\nValidation F1 (weighted):")
    for name, f1 in sorted(val_f1s.items(), key=lambda x: -x[1]):
        print(f"   {name:12s}: {f1:.4f}")
    print(f"\n🏆 Best model by val F1: {best_name}")

    best_model = all_models[best_name]
    test_metrics = evaluate_on_test(best_model, data)

    print("\nTest-set metrics (best model):")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"   {k:12s}: {test_metrics[k]:.4f}")
    print(f"   n_test      : {test_metrics['n_test']}")
    print(f"   positives   : {sum(1 for row in test_metrics['confusion_matrix'] for _ in [row])}")

    # Baseline comparison
    baseline_acc = 0.9951
    baseline_auc = 0.9952
    delta_acc = test_metrics["accuracy"] - baseline_acc
    delta_auc = test_metrics["roc_auc"] - baseline_auc

    print("\nComparison vs. composite-label baseline:")
    print(f"   baseline accuracy (composite label, Apr 2026): {baseline_acc:.4f}")
    print(f"   baseline ROC-AUC  (composite label, Apr 2026): {baseline_auc:.4f}")
    print(f"   this run  accuracy (future-holdout label)    : {test_metrics['accuracy']:.4f}  (Δ = {delta_acc:+.4f})")
    print(f"   this run  ROC-AUC  (future-holdout label)    : {test_metrics['roc_auc']:.4f}  (Δ = {delta_auc:+.4f})")

    print("\nInterpretation:")
    print("   - A smaller ROC-AUC here than the 0.9952 baseline is EXPECTED and")
    print("     desirable — it means the label is no longer reconstructable from")
    print("     step-function thresholds on the same continuous features.")
    print("   - Any residual predictive skill comes from genuine past→future")
    print("     behavioural generalisation on the 4 base features. Adding more")
    print("     behavioural signals (session patterns, workflow n-grams, momentum)")
    print("     would be the next lift.")

    summary = {
        "feature_horizon_days": args.feature_horizon_days,
        "dataset_start": str(dataset_start.date()),
        "split_date": str(split_date.date()),
        "dataset_end": str(dataset_end.date()),
        "n_users": int(n_users),
        "target_positive_rate": float(pos_rate),
        "best_model": best_name,
        "val_f1_per_model": {k: float(v) for k, v in val_f1s.items()},
        "test_metrics": {
            k: (v if isinstance(v, (int, float, list)) else float(v))
            for k, v in test_metrics.items()
        },
        "baseline_comparison": {
            "baseline_accuracy": baseline_acc,
            "baseline_roc_auc": baseline_auc,
            "delta_accuracy": float(delta_acc),
            "delta_roc_auc": float(delta_auc),
        },
        "features_used": FEATURES,
        "label_definition": "user has >=1 event in the target window",
        "notes": (
            "Features computed on feature-window events only; target derived "
            "from a disjoint future window. Model cannot observe the target."
        ),
    }

    print("\nJSON summary:")
    print(json.dumps(summary, indent=2))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n✓ Wrote JSON summary to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
