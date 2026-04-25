import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pickle

print("=" * 80)
print("2. BASE MODELS & ENSEMBLE: TRAINING AND EVALUATION")
print("=" * 80)

# ============================================================================
# BASE MODEL 1: RANDOM FOREST
# ============================================================================
print("\n📊 BASE MODEL 1: RANDOM FOREST")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest (n=200, depth=15, class_weight)...")
rf_model.fit(X_train_resampled, y_train_resampled)

rf_val_pred = rf_model.predict(X_val_scaled)
rf_val_proba = rf_model.predict_proba(X_val_scaled)[:, 1]

rf_val_acc = accuracy_score(y_val, rf_val_pred)
rf_val_f1 = f1_score(y_val, rf_val_pred, average='weighted')

print(f"✓ Random Forest trained - Val Accuracy: {rf_val_acc:.4f}, Val F1: {rf_val_f1:.4f}")

# ============================================================================
# BASE MODEL 2: GRADIENT BOOSTING
# ============================================================================
print("\n📊 BASE MODEL 2: GRADIENT BOOSTING")
print("=" * 80)

gb_model = GradientBoostingClassifier(
    learning_rate=0.05,
    n_estimators=300,
    max_depth=5,
    random_state=42
)

print("Training Gradient Boosting (lr=0.05, n=300)...")
gb_model.fit(X_train_resampled, y_train_resampled)

gb_val_pred = gb_model.predict(X_val_scaled)
gb_val_proba = gb_model.predict_proba(X_val_scaled)[:, 1]

gb_val_acc = accuracy_score(y_val, gb_val_pred)
gb_val_f1 = f1_score(y_val, gb_val_pred, average='weighted')

print(f"✓ Gradient Boosting trained - Val Accuracy: {gb_val_acc:.4f}, Val F1: {gb_val_f1:.4f}")

# ============================================================================
# BASE MODEL 3: ADABOOST
# ============================================================================
print("\n📊 BASE MODEL 3: ADABOOST")
print("=" * 80)

ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

print("Training AdaBoost (n=200, lr=0.05)...")
ada_model.fit(X_train_resampled, y_train_resampled)

ada_val_pred = ada_model.predict(X_val_scaled)
ada_val_proba = ada_model.predict_proba(X_val_scaled)[:, 1]

ada_val_acc = accuracy_score(y_val, ada_val_pred)
ada_val_f1 = f1_score(y_val, ada_val_pred, average='weighted')

print(f"✓ AdaBoost trained - Val Accuracy: {ada_val_acc:.4f}, Val F1: {ada_val_f1:.4f}")

# ============================================================================
# BASE MODEL 4: LOGISTIC REGRESSION
# ============================================================================
print("\n📊 BASE MODEL 4: LOGISTIC REGRESSION (REGULARIZED)")
print("=" * 80)

lr_model = LogisticRegression(
    class_weight=class_weight_dict,
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

print("Training Logistic Regression (L2 regularization, class_weight)...")
lr_model.fit(X_train_resampled, y_train_resampled)

lr_val_pred = lr_model.predict(X_val_scaled)
lr_val_proba = lr_model.predict_proba(X_val_scaled)[:, 1]

lr_val_acc = accuracy_score(y_val, lr_val_pred)
lr_val_f1 = f1_score(y_val, lr_val_pred, average='weighted')

print(f"✓ Logistic Regression trained - Val Accuracy: {lr_val_acc:.4f}, Val F1: {lr_val_f1:.4f}")

# ============================================================================
# ENSEMBLE: SOFT VOTING CLASSIFIER
# ============================================================================
print("\n📊 ENSEMBLE MODEL 1: SOFT VOTING")
print("=" * 80)

# Weight by validation performance (F1 score)
voting_weights = [rf_val_f1, gb_val_f1, ada_val_f1, lr_val_f1]
voting_weights_norm = [w / sum(voting_weights) for w in voting_weights]

print(f"Ensemble weights based on validation F1:")
print(f"   • Random Forest: {voting_weights_norm[0]:.3f}")
print(f"   • Gradient Boosting: {voting_weights_norm[1]:.3f}")
print(f"   • AdaBoost: {voting_weights_norm[2]:.3f}")
print(f"   • Logistic Regression: {voting_weights_norm[3]:.3f}")

voting_ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('ada', ada_model),
        ('lr', lr_model)
    ],
    voting='soft',
    weights=voting_weights_norm
)

print("\nFitting Soft Voting Ensemble...")
voting_ensemble.fit(X_train_resampled, y_train_resampled)

voting_val_pred = voting_ensemble.predict(X_val_scaled)
voting_val_proba = voting_ensemble.predict_proba(X_val_scaled)[:, 1]

voting_val_acc = accuracy_score(y_val, voting_val_pred)
voting_val_f1 = f1_score(y_val, voting_val_pred, average='weighted')

print(f"✓ Soft Voting Ensemble - Val Accuracy: {voting_val_acc:.4f}, Val F1: {voting_val_f1:.4f}")

# ============================================================================
# ENSEMBLE: STACKING WITH META-LEARNER
# ============================================================================
print("\n📊 ENSEMBLE MODEL 2: STACKING")
print("=" * 80)

stacking_ensemble = StackingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('ada', ada_model),
        ('lr', lr_model)
    ],
    final_estimator=LogisticRegression(class_weight=class_weight_dict, max_iter=1000, random_state=42),
    cv=5
)

print("Training Stacking Ensemble with Logistic Regression meta-learner...")
stacking_ensemble.fit(X_train_resampled, y_train_resampled)

stacking_val_pred = stacking_ensemble.predict(X_val_scaled)
stacking_val_proba = stacking_ensemble.predict_proba(X_val_scaled)[:, 1]

stacking_val_acc = accuracy_score(y_val, stacking_val_pred)
stacking_val_f1 = f1_score(y_val, stacking_val_pred, average='weighted')

print(f"✓ Stacking Ensemble - Val Accuracy: {stacking_val_acc:.4f}, Val F1: {stacking_val_f1:.4f}")

# ============================================================================
# VALIDATION RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION RESULTS SUMMARY")
print("=" * 80)

validation_results = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'AdaBoost', 'Logistic Regression', 'Voting Ensemble', 'Stacking Ensemble'],
    'Accuracy': [rf_val_acc, gb_val_acc, ada_val_acc, lr_val_acc, voting_val_acc, stacking_val_acc],
    'F1 Score': [rf_val_f1, gb_val_f1, ada_val_f1, lr_val_f1, voting_val_f1, stacking_val_f1]
})

print("\n📊 Validation Performance:")
print(validation_results.to_string(index=False))

# Select best model based on F1 score
best_idx = validation_results['F1 Score'].idxmax()
best_model_name = validation_results.loc[best_idx, 'Model']
best_f1 = validation_results.loc[best_idx, 'F1 Score']

print(f"\n🏆 Best Model: {best_model_name} (F1: {best_f1:.4f})")

# Map to actual model
model_map = {
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'AdaBoost': ada_model,
    'Logistic Regression': lr_model,
    'Voting Ensemble': voting_ensemble,
    'Stacking Ensemble': stacking_ensemble
}
best_model_obj = model_map[best_model_name]

# ============================================================================
# TEST SET EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("TEST SET EVALUATION")
print("=" * 80)

# Evaluate best model on test set
best_test_pred = best_model_obj.predict(X_test_scaled)
best_test_proba = best_model_obj.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_test, best_test_pred)
test_precision = precision_score(y_test, best_test_pred, average='weighted')
test_recall = recall_score(y_test, best_test_pred, average='weighted')
test_f1 = f1_score(y_test, best_test_pred, average='weighted')
test_roc_auc = roc_auc_score(y_test, best_test_proba)

print(f"\n🏆 {best_model_name} - Test Set Performance:")
print(f"   • Accuracy: {test_acc:.4f}")
print(f"   • Precision: {test_precision:.4f}")
print(f"   • Recall: {test_recall:.4f}")
print(f"   • F1 Score: {test_f1:.4f}")
print(f"   • ROC-AUC: {test_roc_auc:.4f}")

print(f"\n📊 Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, best_test_pred)
print(conf_matrix)
print(f"\n   True Negatives: {conf_matrix[0, 0]}")
print(f"   False Positives: {conf_matrix[0, 1]}")
print(f"   False Negatives: {conf_matrix[1, 0]}")
print(f"   True Positives: {conf_matrix[1, 1]}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

models_to_save = {
    'random_forest': rf_model,
    'gradient_boosting': gb_model,
    'adaboost': ada_model,
    'logistic_regression': lr_model,
    'voting_ensemble': voting_ensemble,
    'stacking_ensemble': stacking_ensemble,
    'best_model': best_model_obj,
    'best_model_name': best_model_name,
    'scaler': scaler_prep,
    'validation_results': validation_results
}

with open('ensemble_models.pkl', 'wb') as f:
    pickle.dump(models_to_save, f)

print("✓ All models saved to 'ensemble_models.pkl'")
print(f"   • Random Forest, Gradient Boosting, AdaBoost, Logistic Regression")
print(f"   • Voting Ensemble, Stacking Ensemble")
print(f"   • Best Model: {best_model_name}")
print(f"   • Scaler for deployment")

print("\n" + "=" * 80)
print("✅ BASE MODELS & ENSEMBLE TRAINING COMPLETE")
print("=" * 80)
