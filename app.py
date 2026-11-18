
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model/student_model.pkl"
METRICS_PATH = "model/metrics.json"

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)


@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

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

        values = {}

        for feature in numeric_features:
            values[feature] = float(data[feature])

        for feature in categorical_features:
            value = str(data[feature]).lower()

            if value not in ["yes", "no"]:
                return jsonify({
                    "error": f"Invalid value for {feature}"
                }), 400

            values[feature] = value

        # Validate numeric ranges
        ranges = {
            "age": (15, 22),
            "studytime": (1, 4),
            "failures": (0, 4),
            "absences": (0, 93),
            "G1": (0, 20),
            "G2": (0, 20),
            "goout": (1, 5),
            "health": (1, 5),
            "freetime": (1, 5)
        }

        for feature, (minimum, maximum) in ranges.items():

            value = values[feature]

            if not value.is_integer():
                return jsonify({
                    "error": f"{feature} must be a whole number"
                }), 400

            if not minimum <= value <= maximum:
                return jsonify({
                    "error": (
                        f"{feature} must be between "
                        f"{minimum} and {maximum}"
                    )
                }), 400

            values[feature] = int(value)

        # Create input DataFrame
        input_df = pd.DataFrame([values])

        # Predict final grade
        prediction = float(model.predict(input_df)[0])

        prediction = max(0, min(20, prediction))

        # Performance category
        if prediction >= 15:
            category = "Excellent"
        elif prediction >= 12:
            category = "Good"
        elif prediction >= 10:
            category = "Average"
        else:
            category = "Needs Improvement"

        return jsonify({
            "prediction": round(prediction, 2),
            "category": category,
            "message": "Prediction completed successfully"
        })

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Please provide all valid student details."
        }), 400

    except Exception as error:
        print("Prediction error:", error)

        return jsonify({
            "error": "Something went wrong while predicting."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
