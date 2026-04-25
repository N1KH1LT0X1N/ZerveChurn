import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Calculate per-user engagement metrics using 'distinct_id' as user identifier
engagement_metrics_grouped = user_retention.groupby('distinct_id').agg({
    'event': 'count',  # total_events
    'timestamp': lambda x: pd.to_datetime(x).dt.date.nunique()  # active_days
}).rename(columns={'event': 'total_events', 'timestamp': 'active_days'})

# Calculate unique event types per user
unique_event_types = user_retention.groupby('distinct_id')['event'].nunique()
engagement_metrics_grouped['unique_event_types'] = unique_event_types

# Calculate session count and events per session
# Define sessions as groups of events within 30 minutes
user_retention_sorted = user_retention.sort_values(['distinct_id', 'timestamp'])
user_retention_sorted['timestamp_dt'] = pd.to_datetime(user_retention_sorted['timestamp'])
user_retention_sorted['time_diff'] = user_retention_sorted.groupby('distinct_id')['timestamp_dt'].diff()
user_retention_sorted['new_session'] = (user_retention_sorted['time_diff'] > pd.Timedelta(minutes=30)) | user_retention_sorted['time_diff'].isna()

session_counts = user_retention_sorted.groupby('distinct_id')['new_session'].sum()
engagement_metrics_grouped['session_count'] = session_counts
engagement_metrics_grouped['avg_events_per_session'] = engagement_metrics_grouped['total_events'] / engagement_metrics_grouped['session_count'].replace(0, 1)

# Handle any missing values
engagement_metrics_grouped = engagement_metrics_grouped.fillna(0)

print(f"Engagement metrics calculated for {len(engagement_metrics_grouped)} users")
print("\nEngagement Metrics Summary:")
print(engagement_metrics_grouped.describe())

# Prepare data for clustering
clustering_features = ['total_events', 'unique_event_types', 'session_count', 'active_days', 'avg_events_per_session']
X_engagement = engagement_metrics_grouped[clustering_features].copy()

# Standardize features for clustering
scaler_engagement = StandardScaler()
X_engagement_scaled = scaler_engagement.fit_transform(X_engagement)

# Apply K-means clustering with k=5
kmeans_engagement = KMeans(n_clusters=5, random_state=42, n_init=20)
engagement_metrics_grouped['cluster'] = kmeans_engagement.fit_predict(X_engagement_scaled)

# Analyze clusters to assign meaningful labels
cluster_profiles = engagement_metrics_grouped.groupby('cluster')[clustering_features].mean().round(2)
print("\nCluster Profiles (Mean Values):")
print(cluster_profiles)

# Assign meaningful labels based on cluster characteristics
def label_engagement_cluster(row):
    total = row['total_events']
    unique = row['unique_event_types']
    sessions = row['session_count']
    days = row['active_days']
    
    if total > 500 and days > 20:
        return 'Power Users'
    elif total > 200 and unique > 30:
        return 'Rising Stars'
    elif total < 20 or days < 3:
        return 'Churned'
    elif sessions > 20 and days > 10:
        return 'Consistent Engagers'
    else:
        return 'Casual Explorers'

cluster_labels_engagement = {}
for _cluster_id in range(5):      # private loop var to avoid conflict
    _cluster_data = cluster_profiles.loc[_cluster_id]   # private loop var
    _label = label_engagement_cluster(_cluster_data)     # private loop var
    cluster_labels_engagement[_cluster_id] = _label

engagement_metrics_grouped['segment_label'] = engagement_metrics_grouped['cluster'].map(cluster_labels_engagement)

print("\nEngagement Segment Distribution:")
print(engagement_metrics_grouped['segment_label'].value_counts())
print("\nSegment Characteristics:")
print(engagement_metrics_grouped.groupby('segment_label')[clustering_features].mean().round(2))

# Store with distinct_id as index for downstream use
engagement_metrics = engagement_metrics_grouped.copy()
