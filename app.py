import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model_path = "lin.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form inputs
        age = float(request.form["age"])
        sex = int(request.form["sex"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = int(request.form["smoker"])
        region = int(request.form["region"])

        # Create array for model prediction
        features = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction=round(prediction, 2),
            form_data=request.form,
        )
    except Exception as e:
        return render_template(
            "index.html", error=str(e), prediction=None
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
