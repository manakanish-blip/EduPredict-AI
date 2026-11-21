
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

# ==================================================
# FILE PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

WITH_GRADES_PATH = os.path.join(
    MODEL_DIR, "student_model_with_grades.pkl"
)

WITHOUT_GRADES_PATH = os.path.join(
    MODEL_DIR, "student_model_without_grades.pkl"
)

METRICS_PATH = os.path.join(
    MODEL_DIR, "metrics.json"
)


# ==================================================
# LOAD BOTH MODELS
# ==================================================

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)

model_with_grades = joblib.load(WITH_GRADES_PATH)
model_without_grades = joblib.load(WITHOUT_GRADES_PATH)


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


# ==================================================
# PERFORMANCE CATEGORY
# ==================================================

def get_category(score):

    if score >= 15:
        return "Excellent"

    elif score >= 12:
        return "Good"

    elif score >= 10:
        return "Average"

    return "Needs Improvement"


# ==================================================
# PERSONALIZED IMPROVEMENT RECOMMENDATIONS
# ==================================================

def generate_recommendations(student):

    recommendations = []
    strengths = []

    # Study time
    if student["studytime"] <= 2:

        recommendations.append({
            "area": "Study Habits",
            "priority": "High",
            "issue": "Limited weekly study time",
            "suggestion": (
                "Create a consistent study schedule. "
                "Gradually increase focused study time "
                "and practice mathematics regularly."
            )
        })

    else:

        strengths.append(
            "Your reported study time is relatively high."
        )

    # Attendance
    if student["absences"] > 10:

        recommendations.append({
            "area": "Attendance",
            "priority": "High",
            "issue": "Higher number of class absences",
            "suggestion": (
                "Try to attend classes regularly and "
                "revise topics missed during absences."
            )
        })

    else:

        strengths.append(
            "Your reported absence count is relatively low."
        )

    # Previous failures
    if student["failures"] > 0:

        recommendations.append({
            "area": "Academic Support",
            "priority": "High",
            "issue": "Previous academic failures reported",
            "suggestion": (
                "Identify difficult topics, seek teacher "
                "guidance and follow a structured revision plan."
            )
        })

    else:

        strengths.append(
            "No previous class failures were reported."
        )

    # Social activities
    if student["goout"] >= 4:

        recommendations.append({
            "area": "Time Management",
            "priority": "Medium",
            "issue": "Frequent social outings reported",
            "suggestion": (
                "Balance social activities with study time "
                "and establish realistic academic goals."
            )
        })

    # Free time
    if student["freetime"] >= 4:

        recommendations.append({
            "area": "Daily Routine",
            "priority": "Medium",
            "issue": "High free-time rating",
            "suggestion": (
                "Consider using some available free time "
                "for revision, homework and practice."
            )
        })

    # Health
    if student["health"] <= 2:

        recommendations.append({
            "area": "Health and Wellbeing",
            "priority": "Medium",
            "issue": "Low self-reported health rating",
            "suggestion": (
                "Maintain healthy routines and seek "
                "appropriate support if health concerns "
                "are affecting your studies."
            )
        })

    # Family support
    if student["famsup"] == "no":

        recommendations.append({
            "area": "Family Support",
            "priority": "Low",
            "issue": "Family educational support not reported",
            "suggestion": (
                "Consider seeking help from teachers, "
                "mentors or available school support services."
            )
        })

    # School support
    if student["schoolsup"] == "no":

        recommendations.append({
            "area": "School Support",
            "priority": "Low",
            "issue": "No additional school support reported",
            "suggestion": (
                "Explore extra academic guidance or "
                "practice sessions if you need them."
            )
        })

    # Paid extra classes
    if student["paid"] == "no":

        recommendations.append({
            "area": "Learning Resources",
            "priority": "Low",
            "issue": "No paid extra classes reported",
            "suggestion": (
                "Use available free learning resources, "
                "textbooks and teacher-provided exercises."
            )
        })

    # Activities
    if student["activities"] == "no":

        recommendations.append({
            "area": "Extracurricular Activities",
            "priority": "Low",
            "issue": "No extracurricular activities reported",
            "suggestion": (
                "Consider a suitable extracurricular activity "
                "that supports a balanced daily routine."
            )
        })

    # Higher education goal
    if student["higher"] == "no":

        recommendations.append({
            "area": "Academic Goals",
            "priority": "Low",
            "issue": "Higher education is not currently planned",
            "suggestion": (
                "Explore your future education and career "
                "options with a teacher or career counsellor."
            )
        })

    # Previous grades (only available in one mode)
    if "G1" in student and "G2" in student:

        if student["G2"] < student["G1"]:

            recommendations.append({
                "area": "Academic Progress",
                "priority": "High",
                "issue": "Second-period grade is lower than first-period grade",
                "suggestion": (
                    "Review topics covered in the second period "
                    "and identify where additional practice is needed."
                )
            })

        elif student["G2"] >= student["G1"]:

            strengths.append(
                "Your second-period grade has maintained "
                "or improved on your first-period grade."
            )

    if not recommendations:

        recommendations.append({
            "area": "Continuous Improvement",
            "priority": "Low",
            "issue": "No major concerns identified by the selected rules",
            "suggestion": (
                "Continue consistent study habits, attend classes "
                "regularly and monitor your academic progress."
            )
        })

    return recommendations, strengths


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        metrics=metrics
    )


# ==================================================
# PREDICTION API
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid request data."
            }), 400

        # Select prediction mode

        prediction_mode = data.get(
            "prediction_mode",
            "with_grades"
        )

        if prediction_mode not in [
            "with_grades",
            "without_grades"
        ]:

            return jsonify({
                "error": "Please select a valid prediction mode."
            }), 400

        # Select features for selected model

        if prediction_mode == "with_grades":

            selected_model = model_with_grades

            numeric_features = numeric_features_with_grades

        else:

            selected_model = model_without_grades

            numeric_features = numeric_features_without_grades

        # Read numeric inputs

        values = {}

        for feature in numeric_features:

            try:

                value = float(data[feature])

            except (KeyError, TypeError, ValueError):

                return jsonify({
                    "error": f"Please provide a valid value for {feature}."
                }), 400

            values[feature] = value

        # Read categorical inputs

        for feature in categorical_features:

            value = str(data.get(feature, "")).lower()

            if value not in ["yes", "no"]:

                return jsonify({
                    "error": f"Invalid value for {feature}."
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

            # Grades are required only in the with-grades mode

            if feature not in values:
                continue

            value = values[feature]

            if not value.is_integer():

                return jsonify({
                    "error": f"{feature} must be a whole number."
                }), 400

            if not minimum <= value <= maximum:

                return jsonify({
                    "error": (
                        f"{feature} must be between "
                        f"{minimum} and {maximum}."
                    )
                }), 400

            values[feature] = int(value)

        # Create model input

        input_df = pd.DataFrame([values])

        # Predict final grade

        prediction = float(
            selected_model.predict(input_df)[0]
        )

        prediction = max(
            0,
            min(20, prediction)
        )

        prediction = round(prediction, 2)

        category = get_category(prediction)

        # Generate improvement recommendations

        recommendations, strengths = generate_recommendations(
            values
        )

        # Model evaluation metrics

        model_key = (
            "with_grades"
            if prediction_mode == "with_grades"
            else "without_grades"
        )

        model_metrics = metrics["models"][model_key]

        return jsonify({

            "prediction": prediction,

            "category": category,

            "prediction_mode": prediction_mode,

            "model_metrics": {
                "mae": model_metrics["mae"],
                "rmse": model_metrics["rmse"],
                "r2": model_metrics["r2"]
            },

            "recommendations": recommendations,

            "strengths": strengths,

            "total_recommendations": len(recommendations),

            "message": "Prediction and analysis completed successfully."

        })

    except Exception as error:

        print("Prediction error:", error)

        return jsonify({
            "error": "Something went wrong while predicting. Please check the input details."
        }), 500


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(debug=True)
