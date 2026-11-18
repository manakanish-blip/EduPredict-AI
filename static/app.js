
const predictionForm = document.getElementById("prediction-form");
const predictButton = document.getElementById("predict-btn");

const resultPlaceholder = document.getElementById("result-placeholder");
const resultContent = document.getElementById("result-content");

const predictedGrade = document.getElementById("predicted-grade");
const performanceCategory = document.getElementById("performance-category");
const predictionMessage = document.getElementById("prediction-message");
const gradeBar = document.getElementById("grade-bar");

const errorMessage = document.getElementById("error-message");


// --------------------------------------------------
// GRADE DISTRIBUTION CHART
// --------------------------------------------------

const chartCanvas = document.getElementById("gradeChart");

if (chartCanvas && typeof gradeDistribution !== "undefined") {

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


// --------------------------------------------------
// PREDICTION FORM
// --------------------------------------------------

predictionForm.addEventListener("submit", async function(event) {

    event.preventDefault();

    // Reset previous errors
    errorMessage.classList.add("hidden");

    // Show loading state
    predictButton.disabled = true;
    predictButton.textContent = "Analyzing Student Data...";

    try {

        const formData = new FormData(predictionForm);

        const studentData = {};

        formData.forEach((value, key) => {
            studentData[key] = value;
        });


        // Send data to Flask backend
        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(studentData)

        });


        const result = await response.json();


        if (!response.ok) {
            throw new Error(
                result.error || "Prediction failed."
            );
        }


        // Display prediction
        const grade = Number(result.prediction);

        resultPlaceholder.classList.add("hidden");
        resultContent.classList.remove("hidden");

        predictedGrade.textContent = grade.toFixed(2);

        performanceCategory.textContent = result.category;

        predictionMessage.textContent =
            "Your predicted final student grade is " +
            grade.toFixed(2) +
            " out of 20.";

        // Animate grade bar
        gradeBar.style.width = "0%";

        requestAnimationFrame(() => {
            gradeBar.style.width = (grade / 20 * 100) + "%";
        });


    } catch (error) {

        console.error("Prediction error:", error);

        errorMessage.textContent = error.message;

        errorMessage.classList.remove("hidden");

    } finally {

        // Restore button
        predictButton.disabled = false;
        predictButton.textContent = "✦ Predict Final Grade";

    }

});
