import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("37100106.csv")

# Predictors
predictors = ['VALUE', 'GEO', 'Age group', 'Type of institution attended', 'REF_DATE']

# Create binary target
median_val = df['VALUE'].median()
df['target_bin'] = (df['VALUE'] > median_val).astype(int)

# Prepare modeling frame
model_df = df[['target_bin'] + predictors].copy()
model_df['VALUE'] = model_df['VALUE'].fillna(model_df['VALUE'].median())
for c in ['GEO','Age group','Type of institution attended','REF_DATE']:
    model_df[c] = model_df[c].astype(str).fillna('MISSING')

# One-hot encode categorical variables
model_enc = pd.get_dummies(model_df, columns=['GEO','Age group','Type of institution attended','REF_DATE'], drop_first=True)

# Split X, y
y = model_enc['target_bin'].astype(int)
X = model_enc.drop(columns=['target_bin'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale numeric
scaler = StandardScaler()
X_train['VALUE'] = scaler.fit_transform(X_train[['VALUE']])
X_test['VALUE'] = scaler.transform(X_test[['VALUE']])

# -------------------------------
# Fit logistic regression
# -------------------------------
clf = LogisticRegression(max_iter=1000, solver='liblinear')
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# -------------------------------
# Print metrics in terminal
# -------------------------------
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
print("Accuracy:", acc)
print("ROC AUC:", roc_auc)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Coefficients
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": clf.coef_.ravel()
}).sort_values(by="coefficient", key=lambda s: s.abs(), ascending=False)
print(coef_df.head(30))

# -------------------------------
# Visualizations (saved as images)
# -------------------------------

# Confusion Matrix Plot
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Below Median", "Above Median"],
            yticklabels=["Below Median", "Above Median"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report Heatmap
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(6,4))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="Greens")
plt.title("Classification Report")
plt.savefig("classification_report.png")
plt.show()

# ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--', color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve.png")
plt.show()
