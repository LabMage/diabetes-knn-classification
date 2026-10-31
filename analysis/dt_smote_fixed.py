from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    average_precision_score, classification_report
)
from imblearn.over_sampling import SMOTE
from common import get_split

# 1. Split first: same test set as kNN 4
X_train, X_test, y_train, y_test = get_split()

# 2. SMOTE only on training data
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(
    X_train, y_train
)

# 3. Train with original hyperparameters
model = DecisionTreeClassifier(max_depth=10, random_state=100)
model.fit(X_train_sm, y_train_sm)

# 4. Evaluate on untouched test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Test class 1 share: {y_test.mean():.4f}")
print(f"SMOTE train class 1 share: {y_train_sm.mean():.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred, digits=3))
