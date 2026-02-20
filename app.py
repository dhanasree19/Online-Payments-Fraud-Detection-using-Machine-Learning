from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("random_forest_model.pkl")


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- INPUT PAGE ----------------
@app.route("/index")
def index():
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    # Get form values
    step = float(request.form["step"])
    type_t = float(request.form["type"])
    amount = float(request.form["amount"])
    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    newbalanceOrig = float(request.form["newbalanceOrig"])
    oldbalanceDest = float(request.form["oldbalanceDest"])
    newbalanceDest = float(request.form["newbalanceDest"])
    isFlaggedFraud = float(request.form["isFlaggedFraud"])

    # Arrange features EXACTLY like training dataset
    data = np.array([[step, type_t, amount,
                      oldbalanceOrg,
                      newbalanceOrig,
                      oldbalanceDest,
                      newbalanceDest,
                      isFlaggedFraud]])

    # Prediction
    pred = model.predict(data)[0]

    # Result message
    if pred == 1:
        result = "⚠️ Fraud Transaction Detected"
    else:
        result = "✅ Safe Transaction"

    return render_template("result.html", prediction=result)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
