import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Prepare time series data per user
user_time_series = []

for user_id in user_retention['uuid'].unique()[:1000]:  # Sample for performance
    user_events = user_retention[user_retention['uuid'] == user_id].copy()
    user_events['timestamp'] = pd.to_datetime(user_events['timestamp'])
    user_events = user_events.sort_values('timestamp')
    
    # Calculate cumulative events over time
    user_events['days_since_start'] = (user_events['timestamp'] - user_events['timestamp'].min()).dt.total_seconds() / (24 * 3600)
    user_events['cumulative_events'] = range(1, len(user_events) + 1)
    
    # Calculate 7-day rolling activity
    user_events['week'] = (user_events['days_since_start'] // 7).astype(int)
    weekly_events = user_events.groupby('week').size().reset_index(name='events_per_week')
    
    user_time_series.append({
        'uuid': user_id,
        'total_events': len(user_events),
        'total_days': user_events['days_since_start'].max(),
        'time_points': user_events['days_since_first_event'].values if 'days_since_first_event' in user_events.columns else user_events['days_since_start'].values,
        'cumulative_events': user_events['cumulative_events'].values,
        'weekly_activity': weekly_events['events_per_week'].values
    })

# Define growth curve models
def exponential_growth(x, a, b, c):
    return a * np.exp(b * x) + c

def linear_growth(x, a, b):
    return a * x + b

def plateau_growth(x, a, b, c):
    return a * (1 - np.exp(-b * x)) + c

def decline_growth(x, a, b, c):
    return a * np.exp(-b * x) + c

# Classify each user's growth trajectory
growth_classifications = []

for user_ts in user_time_series:
    if len(user_ts['time_points']) < 5:  # Need minimum data points
        classification = 'Insufficient_Data'
        best_model = None
        inflection_point = None
    else:
        x = user_ts['time_points']
        y = user_ts['cumulative_events']
        
        # Normalize data
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
        
        # Try fitting different models
        models = {}
        
        # Linear
        try:
            popt, _ = curve_fit(linear_growth, x_norm, y_norm, maxfev=1000)
            y_pred = linear_growth(x_norm, *popt)
            models['Linear'] = np.mean((y_norm - y_pred) ** 2)
        except:
            models['Linear'] = float('inf')
        
        # Exponential
        try:
            popt, _ = curve_fit(exponential_growth, x_norm, y_norm, p0=[1, 0.1, 0], maxfev=1000)
            y_pred = exponential_growth(x_norm, *popt)
            models['Exponential'] = np.mean((y_norm - y_pred) ** 2)
        except:
            models['Exponential'] = float('inf')
        
        # Plateau
        try:
            popt, _ = curve_fit(plateau_growth, x_norm, y_norm, p0=[1, 1, 0], maxfev=1000)
            y_pred = plateau_growth(x_norm, *popt)
            models['Plateau'] = np.mean((y_norm - y_pred) ** 2)
        except:
            models['Plateau'] = float('inf')
        
        # Decline
        try:
            popt, _ = curve_fit(decline_growth, x_norm, y_norm, p0=[1, 0.1, 0], maxfev=1000)
            y_pred = decline_growth(x_norm, *popt)
            models['Decline'] = np.mean((y_norm - y_pred) ** 2)
        except:
            models['Decline'] = float('inf')
        
        # Select best model
        best_model = min(models, key=models.get)
        
        # Detect inflection points (change in velocity)
        if len(user_ts['weekly_activity']) >= 3:
            weekly_diff = np.diff(user_ts['weekly_activity'])
            if len(weekly_diff) > 0:
                max_accel_week = np.argmax(weekly_diff)
                min_accel_week = np.argmin(weekly_diff)
                
                if weekly_diff[max_accel_week] > 0:
                    inflection_point = f"Week {max_accel_week}: Acceleration"
                elif weekly_diff[min_accel_week] < 0:
                    inflection_point = f"Week {min_accel_week}: Deceleration"
                else:
                    inflection_point = "No significant inflection"
            else:
                inflection_point = "No significant inflection"
        else:
            inflection_point = "Insufficient data"
        
        classification = best_model
    
    growth_classifications.append({
        'uuid': user_ts['uuid'],
        'trajectory_type': classification,
        'total_events': user_ts['total_events'],
        'total_days': user_ts['total_days'],
        'inflection_point': inflection_point
    })

growth_trajectory_df = pd.DataFrame(growth_classifications)

print("=" * 70)
print("GROWTH TRAJECTORY CLASSIFICATION")
print("=" * 70)
print(f"\nTotal Users Classified: {len(growth_trajectory_df):,}")
print()

print("Trajectory Distribution:")
print(growth_trajectory_df['trajectory_type'].value_counts().to_string())
print()

print("Average Engagement by Trajectory Type:")
trajectory_summary = growth_trajectory_df.groupby('trajectory_type').agg({
    'total_events': ['mean', 'median'],
    'total_days': ['mean', 'median']
}).round(2)
print(trajectory_summary.to_string())
