from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    average_precision_score, classification_report
)
from common import get_split

X_train, X_test, y_train, y_test = get_split()

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=100
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Test class 1 share: {y_test.mean():.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred, digits=3))
