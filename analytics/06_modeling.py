import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv("analytics/titanic.csv")

X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nSurvival rate in full data:", y.mean().round(4))
print("Survival rate in train:", y_train.mean().round(4))
print("Survival rate in test:", y_test.mean().round(4))

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

log_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
log_reg_pipeline.fit(X_train, y_train)

print(f"\nLogistic Regression — Train accuracy: {log_reg_pipeline.score(X_train, y_train):.4f}")
print(f"Logistic Regression — Test accuracy: {log_reg_pipeline.score(X_test, y_test):.4f}")

dt_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])
dt_pipeline.fit(X_train, y_train)

print(f"\nDecision Tree — Train accuracy: {dt_pipeline.score(X_train, y_train):.4f}")
print(f"Decision Tree — Test accuracy: {dt_pipeline.score(X_test, y_test):.4f}")

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])
rf_pipeline.fit(X_train, y_train)

print(f"\nRandom Forest — Train accuracy: {rf_pipeline.score(X_train, y_train):.4f}")
print(f"Random Forest — Test accuracy: {rf_pipeline.score(X_test, y_test):.4f}")


# Get the actual feature names after one-hot encoding
feature_names = (
    numeric_features +
    list(dt_pipeline.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .named_steps["encoder"]
         .get_feature_names_out(categorical_features))
)

plt.figure(figsize=(20, 10))
plot_tree(
    dt_pipeline.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Died", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=8
)
plt.savefig("analytics/decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved decision_tree.png")




models = {
    "Logistic Regression": log_reg_pipeline,
    "Decision Tree": dt_pipeline,
    "Random Forest": rf_pipeline
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class "1" (survived)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} ---")
    print("Confusion Matrix:")
    print(cm)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "AUC": round(auc, 4)
    })

comparison_df = pd.DataFrame(results)
print("\n=== Model Comparison Table ===")
print(comparison_df.to_string(index=False))


plt.figure(figsize=(6, 5))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All 3 Classifiers")
plt.legend()
plt.savefig("analytics/roc_curves.png")
plt.close()
print("Saved roc_curves.png")



print("\n=== Imbalance Handling Comparison (Random Forest) ===")

# (a) Baseline — already trained as rf_pipeline above, just reuse its results
y_pred_baseline = rf_pipeline.predict(X_test)
print("\n(a) Baseline (no handling):")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.4f}, "
      f"Recall: {recall_score(y_test, y_pred_baseline):.4f}, "
      f"F1: {f1_score(y_test, y_pred_baseline):.4f}")

# (b) class_weight='balanced'
rf_balanced = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)
print("\n(b) class_weight='balanced':")
print(f"Precision: {precision_score(y_test, y_pred_balanced):.4f}, "
      f"Recall: {recall_score(y_test, y_pred_balanced):.4f}, "
      f"F1: {f1_score(y_test, y_pred_balanced):.4f}")

# (c) SMOTE — applied to training fold only, via imblearn's Pipeline
rf_smote = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])
rf_smote.fit(X_train, y_train)
y_pred_smote = rf_smote.predict(X_test)
print("\n(c) SMOTE (train fold only):")
print(f"Precision: {precision_score(y_test, y_pred_smote):.4f}, "
      f"Recall: {recall_score(y_test, y_pred_smote):.4f}, "
      f"F1: {f1_score(y_test, y_pred_smote):.4f}")


rf_for_tuning = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(oob_score=True, random_state=42))
])

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10, 15],
    "classifier__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=rf_for_tuning,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n=== GridSearchCV Results ===")
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation F1 score:", round(grid_search.best_score_, 4))

best_rf = grid_search.best_estimator_
oob_score = best_rf.named_steps["classifier"].oob_score_
print("OOB score of best model:", round(oob_score, 4))

test_acc = best_rf.score(X_test, y_test)
print("Best model — Test accuracy:", round(test_acc, 4))


# Features for predicting fare: everything except fare itself and survived
# (survived is the other task's target, not a legitimate predictor to use here)
reg_features_numeric = ["pclass", "age", "sibsp", "parch"]
reg_features_categorical = ["sex", "embarked"]

X_reg = df[reg_features_numeric + reg_features_categorical]
y_reg = df["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), reg_features_numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), reg_features_categorical)
])

reg_pipeline = Pipeline(steps=[
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression())
])

reg_pipeline.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_pipeline.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)

n = len(y_reg_test)
p = X_reg_test.shape[1]  # number of predictor columns before encoding
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\n=== Fare Regression Results ===")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")



residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(6, 5))
plt.scatter(y_reg_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot — Fare Regression")
plt.savefig("analytics/residual_plot.png")
plt.close()
print("Saved residual_plot.png")



print("\n=== FINAL MODEL COMPARISON ===")

print("\n--- Classification Models ---")
print(comparison_df.to_string(index=False))

regression_summary = pd.DataFrame([{
    "Model": "Linear Regression (fare prediction)",
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "R2": round(r2, 4),
    "Adjusted_R2": round(adjusted_r2, 4)
}])
print("\n--- Regression Model ---")
print(regression_summary.to_string(index=False))


joblib.dump(rf_pipeline, "analytics/best_model_pipeline.joblib")
print("\nSaved best model pipeline to analytics/best_model_pipeline.joblib")

# Reload and verify it works correctly on raw, unpreprocessed data
loaded_pipeline = joblib.load("analytics/best_model_pipeline.joblib")

sample_raw_passenger = X_test.iloc[[0]]
original_prediction = rf_pipeline.predict(sample_raw_passenger)
reloaded_prediction = loaded_pipeline.predict(sample_raw_passenger)

print("Original model prediction:", original_prediction)
print("Reloaded model prediction:", reloaded_prediction)
print("Match:", original_prediction[0] == reloaded_prediction[0])