"""
kNN 4 (corrected): Chi-square feature selection + scaling + SMOTE + kNN.

Fix vs. 2025 version: every step that learns from data (chi2, scaler, SMOTE)
is fitted on the TRAINING set only. The test set is the original, untouched
data with the real class balance.

Run from the repo root:  python analysis/knn4_fixed.py
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, classification_report)
from imblearn.over_sampling import SMOTE

from common import get_split, CONTINUOUS

# Shared split: same untouched test set as all other models
X_train_raw, X_test_raw, y_train_fs, y_test_fs = get_split()

# 1. Chi-square on raw (non-negative) TRAINING data only
selector = SelectKBest(score_func=chi2, k="all").fit(X_train_raw, y_train_fs)
chi2_results = (pd.DataFrame({"Feature": X_train_raw.columns, "Chi2 Score": selector.scores_})
                .sort_values("Chi2 Score", ascending=False))
top_12 = chi2_results.head(12)["Feature"].tolist()

X_train_fs = X_train_raw[top_12].copy()
X_test_fs = X_test_raw[top_12].copy()

# 2. Scale continuous features: fit on train, apply to test
cont_fs = [c for c in CONTINUOUS if c in top_12]
if cont_fs:
    scaler_fs = StandardScaler().fit(X_train_fs[cont_fs])
    X_train_fs[cont_fs] = scaler_fs.transform(X_train_fs[cont_fs])
    X_test_fs[cont_fs] = scaler_fs.transform(X_test_fs[cont_fs])

# 3. SMOTE on TRAINING data only
X_train_fs_sm, y_train_fs_sm = SMOTE(random_state=42).fit_resample(X_train_fs, y_train_fs)

# 4. Train kNN (same hyperparameters as 2025) and evaluate on untouched test set
knn_fs = KNeighborsClassifier(n_neighbors=15, weights="distance", metric="minkowski", p=2)
knn_fs.fit(X_train_fs_sm, y_train_fs_sm)

y_pred_fs = knn_fs.predict(X_test_fs)
y_proba_fs = knn_fs.predict_proba(X_test_fs)[:, 1]

# --- Sanity checks: test set must be real data with ~15% positives ---
print(f"Rows: train={len(X_train_raw)}, test={len(X_test_fs)}")
print(f"Share of class 1 in TEST set: {y_test_fs.mean():.4f}  (expected ~0.153)")
print(f"Share of class 1 in TRAIN after SMOTE: {y_train_fs_sm.mean():.4f}  (expected 0.5)")
print(f"Top 12 features: {top_12}\n")

print("kNN 4 (corrected: Chi2 + SMOTE on train only, k=15, distance weights)")
print(f"Accuracy: {accuracy_score(y_test_fs, y_pred_fs):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test_fs, y_proba_fs):.4f}")
print(f"PR-AUC:   {average_precision_score(y_test_fs, y_proba_fs):.4f}  (random baseline = {y_test_fs.mean():.3f})")
print("\nClassification report (threshold 0.5):")
print(classification_report(y_test_fs, y_pred_fs, digits=3))
