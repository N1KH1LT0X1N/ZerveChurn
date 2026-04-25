import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

print("=" * 80)
print("1. DATA PREPARATION: TRAIN/VAL/TEST SPLIT & CLASS IMBALANCE HANDLING")
print("=" * 80)

# Get success labels and features
success_data = user_success_metrics[['user_id', 'alternative_label']].copy()

# Use alternative_label as it has better balance (see validation block)
# Map to binary: High Value = 1 (success), Growing/Early = 0 (not success)
success_data['target'] = success_data['alternative_label'].map({
    'High Value': 1,
    'Growing': 0,
    'Early/Churned': 0
})

print(f"\n📊 Target Distribution:")
print(f"   • Success (High Value): {(success_data['target'] == 1).sum():,} ({(success_data['target'] == 1).sum() / len(success_data) * 100:.1f}%)")
print(f"   • Not Success: {(success_data['target'] == 0).sum():,} ({(success_data['target'] == 0).sum() / len(success_data) * 100:.1f}%)")

# Merge with user_success_metrics to get features
feature_data = user_success_metrics[[
    'user_id', 'total_events', 'tenure_days', 'days_since_first', 'days_since_last'
]].copy()

# Merge target with features
modeling_data = feature_data.merge(success_data[['user_id', 'target']], on='user_id')

print(f"\n✓ Modeling data shape: {modeling_data.shape}")
print(f"   • Users: {len(modeling_data):,}")
print(f"   • Features: {len(modeling_data.columns) - 2}")  # Exclude user_id and target

# Separate features and target
X = modeling_data.drop(['user_id', 'target'], axis=1)
y = modeling_data['target']
user_ids_full = modeling_data['user_id']

print(f"\n📋 Feature columns: {list(X.columns)}")

# ============================================================================
# STRATIFIED TRAIN/VAL/TEST SPLIT (70/15/15)
# ============================================================================
print("\n" + "=" * 80)
print("STRATIFIED TRAIN/VAL/TEST SPLIT (70/15/15)")
print("=" * 80)

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp, user_train, user_temp = train_test_split(
    X, y, user_ids_full, 
    test_size=0.30, 
    random_state=42, 
    stratify=y
)

# Second split: split temp into validation (50%) and test (50%) = 15% each of original
X_val, X_test, y_val, y_test, user_val, user_test = train_test_split(
    X_temp, y_temp, user_temp,
    test_size=0.50, 
    random_state=42, 
    stratify=y_temp
)

print(f"\n✓ Train set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   • Success: {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
print(f"   • Not Success: {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")

print(f"\n✓ Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"   • Success: {(y_val == 1).sum():,} ({(y_val == 1).sum()/len(y_val)*100:.1f}%)")
print(f"   • Not Success: {(y_val == 0).sum():,} ({(y_val == 0).sum()/len(y_val)*100:.1f}%)")

print(f"\n✓ Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"   • Success: {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
print(f"   • Not Success: {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")

# ============================================================================
# FEATURE SCALING
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE SCALING")
print("=" * 80)

scaler_prep = StandardScaler()
X_train_scaled = scaler_prep.fit_transform(X_train)
X_val_scaled = scaler_prep.transform(X_val)
X_test_scaled = scaler_prep.transform(X_test)

# Convert back to DataFrames for easier handling
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"\n✓ Scaled all features using StandardScaler")
print(f"   • Fitted on training data only")
print(f"   • Applied same transformation to validation and test sets")

# ============================================================================
# HANDLE CLASS IMBALANCE WITH UPSAMPLING (Manual SMOTE alternative)
# ============================================================================
print("\n" + "=" * 80)
print("HANDLING CLASS IMBALANCE WITH UPSAMPLING")
print("=" * 80)

print(f"\n📊 Before balancing:")
print(f"   • Training samples: {len(X_train_scaled):,}")
print(f"   • Success: {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
print(f"   • Not Success: {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print(f"   • Imbalance ratio: 1:{(y_train == 0).sum() / (y_train == 1).sum():.1f}")

# Combine data back for resampling
train_data = X_train_scaled_df.copy()
train_data['target'] = y_train.values

# Separate majority and minority classes
majority_class = train_data[train_data['target'] == 0]
minority_class = train_data[train_data['target'] == 1]

# Upsample minority class
minority_upsampled = resample(minority_class,
                              replace=True,
                              n_samples=len(majority_class),
                              random_state=42)

# Combine and shuffle
train_balanced = pd.concat([majority_class, minority_upsampled])
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X_train_resampled = train_balanced.drop('target', axis=1).values
y_train_resampled = train_balanced['target'].values

print(f"\n✓ After upsampling:")
print(f"   • Training samples: {len(X_train_resampled):,}")
print(f"   • Success: {(y_train_resampled == 1).sum():,} ({(y_train_resampled == 1).sum()/len(y_train_resampled)*100:.1f}%)")
print(f"   • Not Success: {(y_train_resampled == 0).sum():,} ({(y_train_resampled == 0).sum()/len(y_train_resampled)*100:.1f}%)")
print(f"   • Imbalance ratio: 1:{(y_train_resampled == 0).sum() / (y_train_resampled == 1).sum():.1f}")
print(f"   • Samples added: {len(X_train_resampled) - len(X_train_scaled):,}")

# Store class weights for models that support them
n_samples = len(y_train)
n_classes = 2
class_weight_dict = {
    0: n_samples / (n_classes * (y_train == 0).sum()),
    1: n_samples / (n_classes * (y_train == 1).sum())
}

print(f"\n📊 Computed class weights for model training:")
print(f"   • Class 0 (Not Success): {class_weight_dict[0]:.2f}")
print(f"   • Class 1 (Success): {class_weight_dict[1]:.2f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ DATA PREPARATION COMPLETE")
print("=" * 80)

print(f"\n📊 Final Dataset Summary:")
print(f"\nTraining Set (balanced via upsampling):")
print(f"   • Shape: {X_train_resampled.shape}")
print(f"   • Class balance: 50/50")

print(f"\nValidation Set (original distribution):")
print(f"   • Shape: {X_val_scaled.shape}")
print(f"   • Success rate: {(y_val == 1).sum()/len(y_val)*100:.1f}%")

print(f"\nTest Set (original distribution):")
print(f"   • Shape: {X_test_scaled.shape}")
print(f"   • Success rate: {(y_test == 1).sum()/len(y_test)*100:.1f}%")

print(f"\n✅ Ready for model training!")