
// ==================================================
// EDUPREDICT AI - MAIN JAVASCRIPT
// ==================================================

const predictionForm = document.getElementById("prediction-form");
const predictButton = document.getElementById("predict-btn");

// Prediction result elements
const resultPlaceholder = document.getElementById("result-placeholder");
const resultContent = document.getElementById("result-content");

const predictedGrade = document.getElementById("predicted-grade");
const performanceCategory = document.getElementById("performance-category");
const predictionMessage = document.getElementById("prediction-message");
const gradeBar = document.getElementById("grade-bar");

const errorMessage = document.getElementById("error-message");

// Additional result elements
const selectedModel = document.getElementById("selected-model");

const modelMAE = document.getElementById("model-mae");
const modelRMSE = document.getElementById("model-rmse");
const modelR2 = document.getElementById("model-r2");

// Recommendations
const recommendationsPlaceholder =
    document.getElementById("recommendations-placeholder");

const recommendationsContent =
    document.getElementById("recommendations-content");

const recommendationList =
    document.getElementById("recommendation-list");

// Strengths
const strengthsPlaceholder =
    document.getElementById("strengths-placeholder");

const strengthsContent =
    document.getElementById("strengths-content");

const strengthList =
    document.getElementById("strength-list");

// Action plan
const actionPlan = document.getElementById("action-plan");

// Prediction mode
const predictionMode = document.getElementById("prediction-mode");
const gradeFields = document.getElementById("grade-fields");


// ==================================================
// 1. PREDICTION MODE TOGGLE
// ==================================================

function updatePredictionMode() {

    if (!predictionMode || !gradeFields) return;

    const withGrades = predictionMode.value === "with_grades";

    gradeFields.classList.toggle("hidden", !withGrades);

    const g1Input = predictionForm.querySelector('[name="G1"]');
    const g2Input = predictionForm.querySelector('[name="G2"]');

    if (g1Input) {
        g1Input.required = withGrades;
    }

    if (g2Input) {
        g2Input.required = withGrades;
    }
}

if (predictionMode) {
    predictionMode.addEventListener("change", updatePredictionMode);
    updatePredictionMode();
}


// ==================================================
// 2. GRADE DISTRIBUTION CHART
// ==================================================

const chartCanvas = document.getElementById("gradeChart");

if (
    chartCanvas &&
    typeof gradeDistribution !== "undefined" &&
    typeof Chart !== "undefined"
) {

    new Chart(chartCanvas, {

        type: "doughnut",

        data: {
            labels: Object.keys(gradeDistribution),

            datasets: [{
                data: Object.values(gradeDistribution),

                backgroundColor: [
                    "#5795ff",
                    "#a477ff",
                    "#35d69a",
                    "#f5b84b"
                ],

                borderColor: "#111a2e",
                borderWidth: 4,
                hoverOffset: 8
            }]
        },

        options: {

            responsive: true,
            maintainAspectRatio: true,

            plugins: {
                legend: {
                    position: "bottom",

                    labels: {
                        color: "#b5c2d8",
                        padding: 18,
                        usePointStyle: true,

                        font: {
                            size: 10
                        }
                    }
                }
            },

            cutout: "68%"
        }
    });
}


// ==================================================
// 3. DISPLAY MODEL METRICS
// ==================================================

function displayModelMetrics(metrics) {

    if (!metrics) return;

    if (modelMAE) {
        modelMAE.textContent =
            Number(metrics.mae).toFixed(2);
    }

    if (modelRMSE) {
        modelRMSE.textContent =
            Number(metrics.rmse).toFixed(2);
    }

    if (modelR2) {
        modelR2.textContent =
            Number(metrics.r2).toFixed(3);
    }
}


// ==================================================
// 4. DISPLAY RECOMMENDATIONS
// ==================================================

function displayRecommendations(recommendations) {

    if (!recommendationList) return;

    recommendationList.replaceChildren();

    if (!recommendations || recommendations.length === 0) {

        if (recommendationsPlaceholder) {
            recommendationsPlaceholder.classList.remove("hidden");
        }

        if (recommendationsContent) {
            recommendationsContent.classList.add("hidden");
        }

        return;
    }

    if (recommendationsPlaceholder) {
        recommendationsPlaceholder.classList.add("hidden");
    }

    if (recommendationsContent) {
        recommendationsContent.classList.remove("hidden");
    }

    recommendations.forEach((item) => {

        const card = document.createElement("div");
        card.className = "recommendation-card";

        const heading = document.createElement("h4");
        heading.textContent = item.area || "Improvement Area";

        const priority = document.createElement("span");
        priority.className = "recommendation-priority";
        priority.textContent = item.priority || "Recommended";

        const issue = document.createElement("p");
        issue.textContent = item.issue || "";

        const suggestion = document.createElement("p");
        suggestion.textContent = item.suggestion || "";

        card.appendChild(heading);
        card.appendChild(priority);
        card.appendChild(issue);
        card.appendChild(suggestion);

        recommendationList.appendChild(card);
    });
}


// ==================================================
// 5. DISPLAY STUDENT STRENGTHS
// ==================================================

function displayStrengths(strengths) {

    if (!strengthList) return;

    strengthList.replaceChildren();

    if (!strengths || strengths.length === 0) {

        if (strengthsPlaceholder) {
            strengthsPlaceholder.classList.remove("hidden");
        }

        if (strengthsContent) {
            strengthsContent.classList.add("hidden");
        }

        return;
    }

    if (strengthsPlaceholder) {
        strengthsPlaceholder.classList.add("hidden");
    }

    if (strengthsContent) {
        strengthsContent.classList.remove("hidden");
    }

    strengths.forEach((strength) => {

        const item = document.createElement("li");

        item.textContent = strength;

        strengthList.appendChild(item);
    });
}


// ==================================================
// 6. GENERATE PERSONALIZED ACTION PLAN
// ==================================================

function generateActionPlan(recommendations) {

    if (!actionPlan) return;

    actionPlan.replaceChildren();

    if (!recommendations || recommendations.length === 0) {

        const message = document.createElement("p");
        message.textContent =
            "Continue following your current study routine and monitor your progress.";

        actionPlan.appendChild(message);

        return;
    }

    const heading = document.createElement("h4");
    heading.textContent = "Your improvement action plan";

    actionPlan.appendChild(heading);

    recommendations.forEach((item, index) => {

        const planItem = document.createElement("div");
        planItem.className = "action-plan-item";

        const number = document.createElement("strong");
        number.textContent = `${index + 1}. `;

        const description = document.createElement("span");
        description.textContent =
            item.suggestion || item.area || "Review this improvement area.";

        planItem.appendChild(number);
        planItem.appendChild(description);

        actionPlan.appendChild(planItem);
    });
}


// ==================================================
// 7. DISPLAY PREDICTION RESULTS
// ==================================================

function displayPrediction(result) {

    const grade = Number(result.prediction);

    if (!Number.isFinite(grade)) {
        throw new Error("The server returned an invalid prediction.");
    }

    if (resultPlaceholder) {
        resultPlaceholder.classList.add("hidden");
    }

    if (resultContent) {
        resultContent.classList.remove("hidden");
    }

    // Predicted score
    if (predictedGrade) {
        predictedGrade.textContent = grade.toFixed(2);
    }

    // Performance category
    if (performanceCategory) {
        performanceCategory.textContent = result.category || "Not Available";
    }

    // Prediction message
    if (predictionMessage) {
        predictionMessage.textContent =
            result.message ||
            `Your predicted final grade is ${grade.toFixed(2)} out of 20.`;
    }

    // Grade progress bar
    if (gradeBar) {

        gradeBar.style.width = "0%";

        requestAnimationFrame(() => {
            gradeBar.style.width =
                `${Math.max(0, Math.min(100, grade / 20 * 100))}%`;
        });
    }

    // Selected model
    if (selectedModel) {

        selectedModel.textContent =
            result.prediction_mode === "with_grades"
                ? "Model using previous grades"
                : "Model without previous grades";
    }

    // Model performance metrics
    displayModelMetrics(result.model_metrics);

    // Personalized recommendations
    displayRecommendations(result.recommendations);

    // Student strengths
    displayStrengths(result.strengths);

    // Improvement action plan
    generateActionPlan(result.recommendations);
}


// ==================================================
// 8. PREDICTION FORM SUBMISSION
// ==================================================

if (predictionForm) {

    predictionForm.addEventListener("submit", async function(event) {

        event.preventDefault();

        if (errorMessage) {
            errorMessage.classList.add("hidden");
            errorMessage.textContent = "";
        }

        predictButton.disabled = true;
        predictButton.textContent = "Analyzing Student Data...";

        try {

            const formData = new FormData(predictionForm);

            const studentData = {};

            // Collect form values
            formData.forEach((value, key) => {
                studentData[key] = value;
            });

            // Set prediction mode
            studentData.prediction_mode =
                predictionMode ? predictionMode.value : "with_grades";

            // Convert numeric fields
            const numericFields = [
                "age",
                "studytime",
                "failures",
                "absences",
                "G1",
                "G2",
                "goout",
                "health",
                "freetime"
            ];

            numericFields.forEach((field) => {

                if (studentData[field] !== undefined) {
                    studentData[field] = Number(studentData[field]);
                }
            });

            // Remove previous grades when using the second model
            if (studentData.prediction_mode === "without_grades") {
                delete studentData.G1;
                delete studentData.G2;
            }

            // Send data to Flask
            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(studentData)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Prediction failed.");
            }

            // Display complete results
            displayPrediction(result);

        } catch (error) {

            console.error("Prediction error:", error);

            if (errorMessage) {
                errorMessage.textContent =
                    error.message || "Something went wrong. Please try again.";

                errorMessage.classList.remove("hidden");
            }

        } finally {

            predictButton.disabled = false;
            predictButton.textContent = "✦ Predict Final Grade";
        }
    });
}
