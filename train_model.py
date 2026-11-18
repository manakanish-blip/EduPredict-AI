
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "320/student+performance.zip"
)

DATASET_FILE = "student-mat.csv"

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# DOWNLOAD DATASET USING UPDATED SSL CERTIFICATES
# --------------------------------------------------

def download_dataset():

    print("\nDownloading UCI Student Performance dataset...")

    try:
        request = urllib.request.Request(
            DATASET_URL,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
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

            csv_path = None

            for filename in outer_zip.namelist():

                if filename.endswith("student-mat.csv"):
                    csv_path = filename
                    break

                if filename.endswith("student.zip"):

                    nested_data = outer_zip.read(filename)

                    with zipfile.ZipFile(
                        io.BytesIO(nested_data)
                    ) as nested_zip:

                        for nested_filename in nested_zip.namelist():

                            if nested_filename.endswith(
                                "student-mat.csv"
                            ):
                                csv_data = nested_zip.read(
                                    nested_filename
                                )

                                return pd.read_csv(
                                    io.BytesIO(csv_data),
                                    sep=";"
                                )

            if csv_path:

                csv_data = outer_zip.read(csv_path)

                return pd.read_csv(
                    io.BytesIO(csv_data),
                    sep=";"
                )

        raise FileNotFoundError(
            "student-mat.csv was not found inside the downloaded ZIP."
        )

    except Exception as error:

        print("\nDataset download failed!")
        print("Error:", error)

        raise


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

try:

    df = download_dataset()

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

except Exception:

    print("\nCould not download the dataset.")
    print("Please check your internet connection and SSL certificates.")
    raise SystemExit(1)


# --------------------------------------------------
# SELECT FEATURES AND TARGET
# --------------------------------------------------

numeric_features = [
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

categorical_features = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "higher"
]

target = "G3"

required_columns = (
    numeric_features + categorical_features + [target]
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {missing_columns}"
    )

X = df[numeric_features + categorical_features].copy()
y = df[target].copy()


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
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
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# SPLIT DATASET
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n----------- MODEL RESULTS -----------")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

print("------------------------------------")


# --------------------------------------------------
# SAVE TRAINED MODEL
# --------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "student_model.pkl"
)

joblib.dump(model, model_path)

print("\nModel saved:", model_path)


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

metrics = {
    "mae": round(float(mae), 3),
    "rmse": round(float(rmse), 3),
    "r2": round(float(r2), 4),
    "total_students": int(len(df)),
    "training_students": int(len(X_train)),
    "testing_students": int(len(X_test)),
    "grade_distribution": {
        "0-5": int(((df["G3"] >= 0) & (df["G3"] <= 5)).sum()),
        "6-10": int(((df["G3"] >= 6) & (df["G3"] <= 10)).sum()),
        "11-15": int(((df["G3"] >= 11) & (df["G3"] <= 15)).sum()),
        "16-20": int(((df["G3"] >= 16) & (df["G3"] <= 20)).sum())
    }
}

metrics_path = os.path.join(
    MODEL_DIR,
    "metrics.json"
)

with open(metrics_path, "w") as file:

    json.dump(metrics, file, indent=4)

print("Metrics saved:", metrics_path)


# --------------------------------------------------
# COMPLETED
# --------------------------------------------------

print("\n======================================")
print(" STUDENT PERFORMANCE MODEL COMPLETED ")
print("======================================")

print("Dataset records:", len(df))
print("Model:", model_path)
print("Metrics:", metrics_path)

print("\nYou can now proceed to the Flask application!")
