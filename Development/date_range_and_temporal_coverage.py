import pandas as pd

# Identify timestamp columns
timestamp_columns = df.select_dtypes(include=['datetime64']).columns.tolist()

print("=" * 80)
print("TEMPORAL COVERAGE ANALYSIS")
print("=" * 80)
print(f"\nTimestamp Columns Found: {len(timestamp_columns)}")
for col in timestamp_columns:
    print(f"  - {col}")
print()

# Focus on main timestamp column
main_timestamp = 'timestamp'
if main_timestamp in df.columns:
    print("-" * 80)
    print(f"PRIMARY TIMESTAMP: {main_timestamp}")
    print("-" * 80)
    
    # Date range
    min_date = df[main_timestamp].min()
    max_date = df[main_timestamp].max()
    date_range_days = (max_date - min_date).days
    
    print(f"Earliest Event: {min_date}")
    print(f"Latest Event:   {max_date}")
    print(f"Date Range:     {date_range_days} days ({date_range_days/30.44:.1f} months)")
    print()
    
    # Temporal distribution
    print("-" * 80)
    print("TEMPORAL DISTRIBUTION")
    print("-" * 80)
    
    # Events per month
    df_temp = df.copy()
    df_temp['year_month'] = df_temp[main_timestamp].dt.to_period('M')
    monthly_counts = df_temp['year_month'].value_counts().sort_index()
    
    print(f"\nEvents by Month:")
    for period, count in monthly_counts.items():
        print(f"  {period}: {count:,} events ({count/df.shape[0]*100:.1f}%)")
    
    # Events per week day
    df_temp['weekday'] = df_temp[main_timestamp].dt.day_name()
    weekday_counts = df_temp['weekday'].value_counts()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    print(f"\nEvents by Day of Week:")
    for day in weekday_order:
        if day in weekday_counts:
            count = weekday_counts[day]
            print(f"  {day:10s}: {count:,} events ({count/df.shape[0]*100:.1f}%)")
    
    # Check for missing timestamps
    null_timestamps = df[main_timestamp].isnull().sum()
    print(f"\nMissing Timestamps: {null_timestamps:,} ({null_timestamps/df.shape[0]*100:.2f}%)")
