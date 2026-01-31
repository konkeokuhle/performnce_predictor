from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load ML model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_risk_model.pkl")

model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return render_template("educator_form.html")

@app.route("/predict", methods=["POST"])
def predict():
    attendance = int(request.form["attendance"])
    assignment = int(request.form["assignment"])
    test = int(request.form["test"])
    participation = int(request.form["participation"])
    late = int(request.form["late"])

    features = [[attendance, assignment, test, participation, late]]
    prediction = model.predict(features)[0]

    result = "AT RISK" if prediction == 1 else "NOT AT RISK"

    return render_template(
        "educator_form.html",
        prediction=result
    )

if __name__ == "__main__":
    app.run(debug=True)
