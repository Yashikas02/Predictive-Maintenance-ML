# PROJECT: Predictive Maintenance & Failure Early-Warning System
# Dataset: ai4i2020.csv
# Covers all 5 problem statements


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 65)
print("PREDICTIVE MAINTENANCE AND FAILURE EARLY-WARNING SYSTEM")
print("=" * 65)

dataset = pd.read_csv("ai4i2020.csv")

print("\nDataset shape:", dataset.shape)
print("\nFirst five records:")
print(dataset.head())

print("\nMissing values:")
print(dataset.isnull().sum())

target = "Machine failure"

if target not in dataset.columns:
    raise ValueError(
        "Target column 'Machine failure' was not found. "
        "Check the CSV file and column names."
    )

print("\nMachine failure distribution:")
print(dataset[target].value_counts())

# Do not use failure-type indicators as predictors.
# They may reveal the target and cause data leakage.
excluded_columns = [
    "UDI", "Product ID", "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF"
]

feature_columns = [
    col for col in dataset.columns
    if col not in excluded_columns
]

X = dataset[feature_columns].copy()
y = dataset[target].astype(int)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nPredictor columns:", feature_columns)
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Shared train-test split for classification models
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# SHARED PREPROCESSING AND EVALUATION FUNCTIONS
# ============================================================

def make_preprocessor(numeric_cols, categorical_cols):
    """Create preprocessing steps for numeric and categorical data."""

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    return ColumnTransformer(transformers=transformers)


def make_classifier_pipeline(classifier):
    """Combine preprocessing with a classification model."""

    return Pipeline([
        ("preprocessing", make_preprocessor(
            numeric_features, categorical_features
        )),
        ("classifier", classifier)
    ])


all_results = []


def evaluate_model(name, model, X_train_data, X_test_data,
                  y_train_data, y_test_data):
    """Train a model and report its classification performance."""

    print("\n" + "-" * 65)
    print("MODEL:", name)
    print("-" * 65)

    model.fit(X_train_data, y_train_data)

    predictions = model.predict(X_test_data)

    accuracy = accuracy_score(y_test_data, predictions)
    precision = precision_score(
        y_test_data, predictions, zero_division=0
    )
    recall = recall_score(
        y_test_data, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test_data, predictions, zero_division=0
    )

    auc = np.nan

    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test_data)[:, 1]
            auc = roc_auc_score(y_test_data, probabilities)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_data)
            auc = roc_auc_score(y_test_data, scores)
    except (ValueError, AttributeError):
        pass

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    if not np.isnan(auc):
        print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_data, predictions))

    all_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc
    })

    return model


# ============================================================
# PROBLEM 1: MACHINE FAILURE PREDICTION
# Logistic Regression, Decision Tree and SVM
# ============================================================

print("\n\n" + "=" * 65)
print("PROBLEM 1: MACHINE FAILURE PREDICTION")
print("=" * 65)

problem1_models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    )
}

for name, classifier in problem1_models.items():
    pipeline = make_classifier_pipeline(classifier)

    evaluate_model(
        name, pipeline,
        X_train, X_test, y_train, y_test
    )


# ============================================================
# PROBLEM 2: ANALYSIS OF MACHINE OPERATING CONDITIONS
# Exploratory Data Analysis, Correlation and Multiple Regression
# ============================================================

print("\n\n" + "=" * 65)
print("PROBLEM 2: ANALYSIS OF OPERATING CONDITIONS")
print("=" * 65)

print("\nNumerical summary:")
print(dataset[numeric_features].describe())

print("\nCorrelation matrix:")
correlation_matrix = dataset[numeric_features].corr()
print(correlation_matrix.round(3))

plt.figure(figsize=(10, 7))
plt.imshow(correlation_matrix, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)
plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)
plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.show()

# Multiple Linear Regression:
# Predict torque from other operating conditions.
regression_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Tool wear [min]"
]
regression_target = "Torque [Nm]"

required_columns = regression_features + [regression_target]

missing_columns = [
    col for col in required_columns
    if col not in dataset.columns
]

if missing_columns:
    print("\nSkipping regression; missing columns:", missing_columns)
else:
    regression_data = dataset[required_columns].copy()

    X_reg = regression_data[regression_features]
    y_reg = regression_data[regression_target]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
        train_test_split(
            X_reg, y_reg,
            test_size=0.20,
            random_state=42
        )
    )

    regression_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    regression_pipeline.fit(X_reg_train, y_reg_train)
    reg_predictions = regression_pipeline.predict(X_reg_test)

    mae = mean_absolute_error(y_reg_test, reg_predictions)
    mse = mean_squared_error(y_reg_test, reg_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_reg_test, reg_predictions)

    print("\nMultiple Linear Regression Results")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_reg_test, reg_predictions, alpha=0.5)
    plt.xlabel("Actual Torque [Nm]")
    plt.ylabel("Predicted Torque [Nm]")
    plt.title("Actual vs Predicted Torque")
    plt.tight_layout()
    plt.show()


# ============================================================
# PROBLEM 3: MACHINE OPERATING CONDITION GROUPING
# K-Means and Hierarchical Clustering
# ============================================================

print("\n\n" + "=" * 65)
print("PROBLEM 3: MACHINE OPERATING CONDITION GROUPING")
print("=" * 65)

cluster_features = [
    col for col in numeric_features
    if col in dataset.columns
]

cluster_data = dataset[cluster_features].copy()

cluster_imputer = SimpleImputer(strategy="median")
cluster_scaler = StandardScaler()

cluster_array = cluster_imputer.fit_transform(cluster_data)
cluster_array = cluster_scaler.fit_transform(cluster_array)

# K-Means clustering
kmeans = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(cluster_array)

print("\nK-Means cluster sizes:")
print(pd.Series(kmeans_labels).value_counts().sort_index())

sample_size = min(2000, len(cluster_array))

print(
    "K-Means silhouette score:",
    round(
        silhouette_score(
            cluster_array,
            kmeans_labels,
            sample_size=sample_size,
            random_state=42
        ),
        4
    )
)

kmeans_results = pd.DataFrame({
    "Cluster": kmeans_labels,
    "Machine failure": y.to_numpy()
})

print("\nFailure rate by K-Means cluster:")
print(kmeans_results.groupby("Cluster")["Machine failure"].mean())

plt.figure(figsize=(8, 5))
plt.scatter(
    dataset["Rotational speed [rpm]"],
    dataset["Torque [Nm]"],
    c=kmeans_labels,
    alpha=0.6
)
plt.xlabel("Rotational speed [rpm]")
plt.ylabel("Torque [Nm]")
plt.title("K-Means Clusters of Operating Conditions")
plt.tight_layout()
plt.show()

# Hierarchical clustering on a sample to reduce computation time
hierarchical_sample_size = min(1500, len(dataset))

rng = np.random.RandomState(42)
sample_indices = rng.choice(
    len(dataset),
    size=hierarchical_sample_size,
    replace=False
)

hierarchical_data = cluster_array[sample_indices]
hierarchical_failures = y.iloc[sample_indices].to_numpy()

hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(hierarchical_data)

print("\nHierarchical cluster sizes:")
print(pd.Series(hierarchical_labels).value_counts().sort_index())

if len(np.unique(hierarchical_labels)) > 1:
    print(
        "Hierarchical silhouette score:",
        round(
            silhouette_score(
                hierarchical_data, hierarchical_labels
            ),
            4
        )
    )

hierarchical_results = pd.DataFrame({
    "Cluster": hierarchical_labels,
    "Machine failure": hierarchical_failures
})

print("\nFailure rate by hierarchical cluster:")
print(
    hierarchical_results.groupby("Cluster")["Machine failure"].mean()
)


# ============================================================
# PROBLEM 4: COMPARISON OF FAILURE PREDICTION MODELS
# Decision Tree, Random Forest, Bagging and Boosting
# ============================================================

print("\n\n" + "=" * 65)
print("PROBLEM 4: MODEL COMPARISON AND ENSEMBLE LEARNING")
print("=" * 65)

problem4_models = {
    "Decision Tree - Comparison": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ),

    "Bagging": BaggingClassifier(
        n_estimators=10,
        random_state=42,
        n_jobs=1
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=30,
        random_state=42
    )
}

for name, classifier in problem4_models.items():
    pipeline = make_classifier_pipeline(classifier)

    evaluate_model(
        name, pipeline,
        X_train, X_test, y_train, y_test
    )


# ============================================================
# PROBLEM 5: NEURAL NETWORK-BASED FAILURE PREDICTION
# MLP and PCA + MLP
# ============================================================

print("\n\n" + "=" * 65)
print("PROBLEM 5: NEURAL NETWORK FAILURE PREDICTION")
print("=" * 65)

# Standard MLP classifier
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    max_iter=300,
    early_stopping=True,
    random_state=42
)

mlp_pipeline = make_classifier_pipeline(mlp_classifier)

evaluate_model(
    "Neural Network (MLP)", mlp_pipeline,
    X_train, X_test, y_train, y_test
)

# PCA + MLP using numerical features
X_train_numeric = X_train[numeric_features].copy()
X_test_numeric = X_test[numeric_features].copy()

pca_mlp_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=300,
        early_stopping=True,
        random_state=42
    ))
])

evaluate_model(
    "PCA + Neural Network",
    pca_mlp_pipeline,
    X_train_numeric, X_test_numeric, y_train, y_test
)

print("\nPCA components retained:",
      pca_mlp_pipeline.named_steps["pca"].n_components_)
print("Total variance retained:",
      round(
          pca_mlp_pipeline.named_steps["pca"]
          .explained_variance_ratio_.sum(),
          4
      ))


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print("\n\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
).reset_index(drop=True)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "machine_failure_model_comparison.csv",
    index=False
)

print("\nComparison table saved as:")
print("machine_failure_model_comparison.csv")

# Plot F1-scores for all classification models
plt.figure(figsize=(12, 6))
plt.bar(results_df["Model"], results_df["F1-score"])
plt.xlabel("Model")
plt.ylabel("F1-score")
plt.title("Comparison of Machine Failure Prediction Models")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Display the highest F1-score model
best_model = results_df.iloc[0]

print("\nBest model based on F1-score:")
print("Model:", best_model["Model"])
print("F1-score:", round(best_model["F1-score"], 4))
print("Recall:", round(best_model["Recall"], 4))
print("Precision:", round(best_model["Precision"], 4))

print("\nAll five problem statements have been processed.")
print("Please interpret the results using the printed metrics.")
