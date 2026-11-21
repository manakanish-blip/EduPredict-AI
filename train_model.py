
import os
import ssl
import json
import zipfile
import urllib.request
import io

import certifi
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# CONFIGURATION
# ==================================================

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "320/student+performance.zip"
)

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==================================================
# DOWNLOAD UCI DATASET
# ==================================================

def download_dataset():

    print("\nDownloading UCI Student Performance dataset...")

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    ssl_context = ssl.create_default_context(
        cafile=certifi.where()
    )

    with urllib.request.urlopen(
        request,
        context=ssl_context,
        timeout=60
    ) as response:

        zip_data = response.read()

    print("Dataset downloaded successfully!")

    with zipfile.ZipFile(io.BytesIO(zip_data)) as outer_zip:

        for filename in outer_zip.namelist():

            if filename.endswith("student-mat.csv"):

                return pd.read_csv(
                    io.BytesIO(outer_zip.read(filename)),
                    sep=";"
                )

            if filename.endswith("student.zip"):

                nested_data = outer_zip.read(filename)

                with zipfile.ZipFile(
                    io.BytesIO(nested_data)
                ) as nested_zip:

                    for nested_filename in nested_zip.namelist():

                        if nested_filename.endswith("student-mat.csv"):

                            return pd.read_csv(
                                io.BytesIO(
                                    nested_zip.read(nested_filename)
                                ),
                                sep=";"
                            )

    raise FileNotFoundError(
        "student-mat.csv was not found in the ZIP file."
    )


# ==================================================
# LOAD DATASET
# ==================================================

try:

    df = download_dataset()

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

except Exception as error:

    print("\nDataset download failed!")
    print("Error:", error)

    raise SystemExit(1)


# ==================================================
# FEATURE CONFIGURATION
# ==================================================

numeric_features_with_grades = [
    "age",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2",
    "goout",
    "health",
    "freetime"
]

numeric_features_without_grades = [
    "age",
    "studytime",
    "failures",
    "absences",
    "goout",
    "health",
    "freetime"
]

categorical_features = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "higher"
]

target = "G3"


# ==================================================
# VALIDATE DATASET
# ==================================================

required_columns = (
    numeric_features_with_grades
    + categorical_features
    + [target]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

df = df.dropna(subset=[target]).copy()

print("\nDataset validation completed.")


# ==================================================
# PREPROCESSING FUNCTION
# ==================================================

def create_preprocessor(numeric_features):

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor


# ==================================================
# RANDOM FOREST MODEL
# ==================================================

def create_model(numeric_features):

    preprocessor = create_preprocessor(
        numeric_features
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),

            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=4,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    return model


# ==================================================
# TRAINING AND EVALUATION FUNCTION
# ==================================================

def train_and_evaluate(
    model_name,
    numeric_features,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n===================================")
    print("Training:", model_name)
    print("===================================")

    all_features = (
        numeric_features + categorical_features
    )

    model = create_model(numeric_features)

    model.fit(
        X_train[all_features],
        y_train
    )

    predictions = model.predict(
        X_test[all_features]
    )

    predictions = np.clip(
        predictions,
        0,
        20
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nModel evaluation:")

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.4f}")

    # Save model

    model_filename = (
        "student_model_with_grades.pkl"
        if "WITHOUT" not in model_name
        else "student_model_without_grades.pkl"
    )

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved:", model_path)

    # Feature importance

    regressor = model.named_steps["regressor"]

    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    importance_values = regressor.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\nTop 5 influential model features:")

    print(
        feature_importance.head(5).to_string(index=False)
    )

    return {
        "model_path": model_filename,
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 4),
        "top_features": [
            {
                "feature": str(row["feature"]),
                "importance": round(
                    float(row["importance"]),
                    5
                )
            }
            for _, row in feature_importance.head(10).iterrows()
        ]
    }


# ==================================================
# SPLIT DATASET
# ==================================================

X = df.drop(columns=[target])

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==================================================
# TRAIN MODEL 1: WITH PREVIOUS GRADES
# ==================================================

results_with_grades = train_and_evaluate(
    "MODEL WITH PREVIOUS GRADES",
    numeric_features_with_grades,
    X_train,
    X_test,
    y_train,
    y_test
)


# ==================================================
# TRAIN MODEL 2: WITHOUT PREVIOUS GRADES
# ==================================================

results_without_grades = train_and_evaluate(
    "MODEL WITHOUT PREVIOUS GRADES",
    numeric_features_without_grades,
    X_train,
    X_test,
    y_train,
    y_test
)


# ==================================================
# GRADE DISTRIBUTION
# ==================================================

grade_distribution = {
    "0-5": int(
        ((df["G3"] >= 0) & (df["G3"] <= 5)).sum()
    ),

    "6-10": int(
        ((df["G3"] >= 6) & (df["G3"] <= 10)).sum()
    ),

    "11-15": int(
        ((df["G3"] >= 11) & (df["G3"] <= 15)).sum()
    ),

    "16-20": int(
        ((df["G3"] >= 16) & (df["G3"] <= 20)).sum()
    )
}


# ==================================================
# SAVE MODEL METRICS
# ==================================================

metrics = {

    "total_students": int(len(df)),

    "training_students": int(len(X_train)),

    "testing_students": int(len(X_test)),

    "grade_distribution": grade_distribution,

    "models": {

        "with_grades": results_with_grades,

        "without_grades": results_without_grades

    }

}

metrics_path = os.path.join(
    MODEL_DIR,
    "metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==================================================
# COMPLETED
# ==================================================

print("\n======================================")
print(" EDUPREDICT AI MODEL TRAINING COMPLETE")
print("======================================")

print("\nBoth prediction models have been trained!")

print("\nGenerated files:")

print("1. model/student_model_with_grades.pkl")

print("2. model/student_model_without_grades.pkl")

print("3. model/metrics.json")

print("\nYou can now proceed to the upgraded Flask application!")
