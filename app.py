import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model safely
MODEL_PATH = "lin.pkl"
model = None

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Cost Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: var(--card-bg);
            width: 100%;
            max-width: 680px;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 28px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 8px;
        }

        .header p {
            font-size: 14px;
            color: var(--text-muted);
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            font-weight: 600;
            font-size: 13px;
            color: #374151;
            margin-bottom: 8px;
        }

        input, select {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            font-size: 14px;
            color: #1f2937;
            background-color: #f9fafb;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            outline: none;
            background-color: #fff;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
        }

        .btn-submit {
            width: 100%;
            background: var(--primary);
            color: white;
            border: none;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.1s ease;
            margin-top: 10px;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
        }

        .btn-submit:hover {
            background: var(--primary-hover);
            transform: translateY(-1px);
        }

        .result-card {
            margin-top: 30px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 24px;
            border-radius: 14px;
            text-align: center;
            animation: fadeIn 0.4s ease-in-out;
            box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25);
        }

        .result-card h3 {
            font-size: 13px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
            opacity: 0.9;
        }

        .result-card .price {
            font-size: 32px;
            font-weight: 700;
        }

        .error-card {
            margin-top: 20px;
            background: #fee2e2;
            border: 1px solid #f87171;
            color: #b91c1c;
            padding: 15px;
            border-radius: 10px;
            font-size: 14px;
            text-align: center;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
                gap: 0;
            }
            .container {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical Insurance Cost Predictor</h1>
            <p>Fill out the details below to instantly calculate your estimated medical expenses.</p>
        </div>

        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" min="1" max="120" value="{{ form_data.age if form_data else '30' }}" required>
                </div>

                <div class="form-group">
                    <label for="sex">Sex</label>
                    <select id="sex" name="sex">
                        <option value="0" {% if form_data and form_data.sex == '0' %}selected{% endif %}>Female</option>
                        <option value="1" {% if form_data and form_data.sex == '1' %}selected{% endif %}>Male</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="bmi">BMI (Body Mass Index)</label>
                    <input type="number" step="0.1" id="bmi" name="bmi" min="10" max="60" value="{{ form_data.bmi if form_data else '25.0' }}" required>
                </div>

                <div class="form-group">
                    <label for="children">Number of Children</label>
                    <input type="number" id="children" name="children" min="0" max="10" value="{{ form_data.children if form_data else '0' }}" required>
                </div>

                <div class="form-group">
                    <label for="smoker">Smoker Status</label>
                    <select id="smoker" name="smoker">
                        <option value="0" {% if form_data and form_data.smoker == '0' %}selected{% endif %}>No</option>
                        <option value="1" {% if form_data and form_data.smoker == '1' %}selected{% endif %}>Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="region">Region</label>
                    <select id="region" name="region">
                        <option value="0" {% if form_data and form_data.region == '0' %}selected{% endif %}>Northeast</option>
                        <option value="1" {% if form_data and form_data.region == '1' %}selected{% endif %}>Northwest</option>
                        <option value="2" {% if form_data and form_data.region == '2' %}selected{% endif %}>Southeast</option>
                        <option value="3" {% if form_data and form_data.region == '3' %}selected{% endif %}>Southwest</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="btn-submit">Calculate Insurance Cost</button>
        </form>

        {% if error %}
        <div class="error-card">
            <strong>Error:</strong> {{ error }}
        </div>
        {% endif %}

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Estimated Annual Cost</h3>
            <div class="price">${{ "{:,.2f}".format(prediction) }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    form_data = None

    if request.method == "POST":
        try:
            form_data = request.form
            age = float(form_data["age"])
            sex = int(form_data["sex"])
            bmi = float(form_data["bmi"])
            children = int(form_data["children"])
            smoker = int(form_data["smoker"])
            region = int(form_data["region"])

            if model is None:
                raise Exception(
                    "Model file could not be loaded on the server. Verify that 'lin.pkl' is uploaded in your root directory."
                )

            features = np.array([[age, sex, bmi, children, smoker, region]])
            
            # Fixed: Safely flatten the array output to extract the scalar value cleanly
            raw_prediction = model.predict(features)
            prediction = float(np.ravel(raw_prediction)[0])

        except Exception as e:
            error = str(e)
            print(f"Prediction error details: {e}")

    return render_template_string(
        HTML_TEMPLATE, prediction=prediction, error=error, form_data=form_data
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
