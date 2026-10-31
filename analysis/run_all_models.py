"""
Corrected evaluation of all 7 configurations from the 2025 capstone.

- One shared split (common.get_split): duplicates removed, stratified 70/30, random_state=100.
- Every step that learns from data (scalers, chi2, SMOTE) is fitted on TRAIN only.
- All models are evaluated on the same untouched test set (real class balance).
- Hyperparameters are kept exactly as in the 2025 version (no new tuning).

Outputs (in analysis/):
  results.csv     - metrics table for all models (source for Tables 3 and 4)
  roc_curve.png   - ROC curves of all models on the same test set (Figure 10)

Run from the repo root:  python analysis/run_all_models.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score, roc_curve)
from imblearn.over_sampling import SMOTE

from common import get_split, CONTINUOUS

OUT = os.path.dirname(os.path.abspath(__file__))
X_train, X_test, y_train, y_test = get_split()
results, roc_data = [], {}


def evaluate(name, model, Xtr, ytr, Xte, settings):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    results.append({
        "Model": name, **settings,
        "Accuracy": accuracy_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "PR_AUC": average_precision_score(y_test, proba),
        "Precision_1": precision_score(y_test, pred),
        "Recall_1": recall_score(y_test, pred),
        "F1_1": f1_score(y_test, pred),
    })
    roc_data[name] = roc_curve(y_test, proba)[:2] + (results[-1]["ROC_AUC"],)
    print(f"{name:28s} ROC-AUC={results[-1]['ROC_AUC']:.3f}  PR-AUC={results[-1]['PR_AUC']:.3f}  "
          f"Recall_1={results[-1]['Recall_1']:.3f}")


def scale_continuous(Xtr, Xte, cols):
    """StandardScaler on given continuous columns: fit on train, apply to test."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    if cols:
        sc = StandardScaler().fit(Xtr[cols])
        Xtr[cols], Xte[cols] = sc.transform(Xtr[cols]), sc.transform(Xte[cols])
    return Xtr, Xte


# kNN 1: baseline, StandardScaler on all features, k=5
sc = StandardScaler().fit(X_train)
evaluate("kNN 1 (baseline)", KNeighborsClassifier(n_neighbors=5),
         sc.transform(X_train), y_train, sc.transform(X_test),
         {"k": 5, "Distance": "Euclidean", "Weights": "Uniform", "Scaler": "Standard (all)",
          "SMOTE": "No", "Feature_selection": "No"})

# kNN 2: RobustScaler on all features, k=15, Manhattan, distance weights
rs = RobustScaler().fit(X_train)
evaluate("kNN 2 (RobustScaler)", KNeighborsClassifier(n_neighbors=15, weights="distance", p=1),
         rs.transform(X_train), y_train, rs.transform(X_test),
         {"k": 15, "Distance": "Manhattan", "Weights": "Distance", "Scaler": "Robust (all)",
          "SMOTE": "No", "Feature_selection": "No"})

# kNN 3: scale continuous features, SMOTE on train, k=10
Xtr3, Xte3 = scale_continuous(X_train, X_test, CONTINUOUS)
Xtr3_sm, ytr3_sm = SMOTE(random_state=42).fit_resample(Xtr3, y_train)
evaluate("kNN 3 (SMOTE)", KNeighborsClassifier(n_neighbors=10),
         Xtr3_sm, ytr3_sm, Xte3,
         {"k": 10, "Distance": "Euclidean", "Weights": "Uniform", "Scaler": "Standard (continuous)",
          "SMOTE": "Yes (train only)", "Feature_selection": "No"})

# kNN 4: chi2 top-12 on raw train, scale continuous, SMOTE on train, k=15, distance weights
chi = SelectKBest(chi2, k="all").fit(X_train, y_train)
chi2_results = (pd.DataFrame({"Feature": X_train.columns, "Chi2 Score": chi.scores_})
                .sort_values("Chi2 Score", ascending=False))
top_12 = chi2_results.head(12)["Feature"].tolist()
Xtr4, Xte4 = scale_continuous(X_train[top_12], X_test[top_12],
                              [c for c in CONTINUOUS if c in top_12])
Xtr4_sm, ytr4_sm = SMOTE(random_state=42).fit_resample(Xtr4, y_train)
evaluate("kNN 4 (Chi2 + SMOTE)", KNeighborsClassifier(n_neighbors=15, weights="distance"),
         Xtr4_sm, ytr4_sm, Xte4,
         {"k": 15, "Distance": "Euclidean", "Weights": "Distance", "Scaler": "Standard (continuous)",
          "SMOTE": "Yes (train only)", "Feature_selection": "Chi2 top 12 (train only)"})

# Decision Tree, imbalanced (trees need no scaling)
evaluate("Decision Tree", DecisionTreeClassifier(max_depth=10, random_state=100),
         X_train, y_train, X_test,
         {"k": None, "Distance": None, "Weights": None, "Scaler": "None",
          "SMOTE": "No", "Feature_selection": "No"})

# Decision Tree, SMOTE on train only
Xtr_sm, ytr_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)
evaluate("Decision Tree (SMOTE)", DecisionTreeClassifier(max_depth=10, random_state=100),
         Xtr_sm, ytr_sm, X_test,
         {"k": None, "Distance": None, "Weights": None, "Scaler": "None",
          "SMOTE": "Yes (train only)", "Feature_selection": "No"})

# Random Forest, imbalanced
evaluate("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=15,
                                                 random_state=100, n_jobs=-1),
         X_train, y_train, X_test,
         {"k": None, "Distance": None, "Weights": None, "Scaler": "None",
          "SMOTE": "No", "Feature_selection": "No"})

# --- Save metrics table ---
res = pd.DataFrame(results)
metric_cols = ["Accuracy", "ROC_AUC", "PR_AUC", "Precision_1", "Recall_1", "F1_1"]
res[metric_cols] = res[metric_cols].round(3)
res.to_csv(os.path.join(OUT, "results.csv"), index=False)
chi2_results.to_csv(os.path.join(OUT, "chi2_scores_train.csv"), index=False)
print(f"\nTest set: {len(y_test)} rows, class 1 share = {y_test.mean():.4f}")
print(res[["Model"] + metric_cols].to_string(index=False))

# --- ROC curves on the same test set (save BEFORE closing) ---
fig, ax = plt.subplots(figsize=(8, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 10: ROC Curves (all models, same test set)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "roc_curve.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
