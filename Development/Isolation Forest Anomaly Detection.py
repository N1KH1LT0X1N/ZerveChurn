import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("=" * 80)
print("ISOLATION FOREST: DETECTING EXCEPTIONAL USER BEHAVIORS")
print("=" * 80)

# Prepare feature matrix from behavioral fingerprint
features_for_anomaly = [
    'total_sessions', 'avg_session_length', 'max_session_length',
    'avg_events_per_session', 'total_events', 'avg_event_density',
    'deep_work_sessions', 'deep_work_ratio', 'sequence_diversity',
    'power_user_score', 'struggle_score', 'error_count',
    'trigram_count', 'collaboration_ratio'
]

# Select available features
available_features_anomaly = [f for f in features_for_anomaly if f in behavioral_fingerprint.columns]
print(f"\n✓ Using {len(available_features_anomaly)} behavioral features for anomaly detection")

anomaly_data = behavioral_fingerprint[['user_id'] + available_features_anomaly].copy()

# Handle any remaining NaN
anomaly_data.fillna(0, inplace=True)

# Separate features
X_anomaly = anomaly_data[available_features_anomaly]

# Standardize features
scaler_anomaly = StandardScaler()
X_scaled_anomaly = scaler_anomaly.fit_transform(X_anomaly)

print("\n" + "=" * 80)
print("TRAINING ISOLATION FOREST MODEL")
print("=" * 80)

# Train Isolation Forest
# contamination parameter: expected proportion of outliers (use 5% for top 5%)
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # Detect top 5% as outliers
    random_state=42,
    n_jobs=-1,
    max_samples='auto'
)

print("\nTraining Isolation Forest (n_estimators=200, contamination=0.05)...")
iso_forest.fit(X_scaled_anomaly)

# Predict anomalies (-1 for outliers, 1 for inliers)
anomaly_predictions = iso_forest.predict(X_scaled_anomaly)
anomaly_scores = iso_forest.score_samples(X_scaled_anomaly)  # Lower = more anomalous

# Add predictions to data
anomaly_data['anomaly_prediction'] = anomaly_predictions
anomaly_data['anomaly_score'] = anomaly_scores
anomaly_data['is_exceptional'] = (anomaly_predictions == -1).astype(int)

exceptional_users_count = anomaly_data['is_exceptional'].sum()
print(f"\n✓ Model trained successfully")
print(f"✓ Identified {exceptional_users_count:,} exceptional users ({exceptional_users_count/len(anomaly_data)*100:.1f}%)")

# Get exceptional users
exceptional_users_df = anomaly_data[anomaly_data['is_exceptional'] == 1].copy()
exceptional_users_df = exceptional_users_df.sort_values('anomaly_score', ascending=True)

print("\n" + "=" * 80)
print("EXCEPTIONAL USERS PROFILE")
print("=" * 80)

print("\nTop 10 Most Exceptional Users (by anomaly score):")
print(exceptional_users_df[['user_id', 'anomaly_score', 'total_events', 'total_sessions', 
                              'power_user_score', 'deep_work_ratio']].head(10).to_string(index=False))

# Compare exceptional vs normal users
print("\n" + "=" * 80)
print("EXCEPTIONAL VS NORMAL USER COMPARISON")
print("=" * 80)

normal_users_df = anomaly_data[anomaly_data['is_exceptional'] == 0]

comparison_features = ['total_events', 'total_sessions', 'avg_session_length', 
                       'power_user_score', 'deep_work_ratio', 'sequence_diversity']
comparison_stats = pd.DataFrame({
    'Exceptional (Mean)': exceptional_users_df[comparison_features].mean(),
    'Normal (Mean)': normal_users_df[comparison_features].mean(),
    'Ratio (Exceptional/Normal)': exceptional_users_df[comparison_features].mean() / normal_users_df[comparison_features].mean()
})

print("\n", comparison_stats.round(2))

print("\n" + "=" * 80)
print("✅ ISOLATION FOREST ANOMALY DETECTION COMPLETE")
print("=" * 80)
