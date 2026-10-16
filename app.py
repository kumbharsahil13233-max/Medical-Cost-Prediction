import pickle
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor", page_icon="🏥", layout="centered"
)

# Custom CSS for styling and layout
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5a6fe8 0%, #6a4196 100%);
        color: white;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-top: 25px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained model
@st.cache_resource
def load_model():
    with open("lin.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# App Header
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown(
    "Provide your details below to instantly estimate your annual medical insurance charges using a machine learning linear regression model."
)
st.markdown("---")

# Layout using columns for a clean form interface
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
    )
    children = st.number_input(
        "Number of Children", min_value=0, max_value=10, value=0, step=1
    )

with col2:
    sex = st.selectbox("Sex", ["Female", "Male"])
    smoker = st.selectbox("Smoker Status", ["No", "Yes"])
    region = st.selectbox(
        "Region", ["Northeast", "Northwest", "Southeast", "Southwest"]
    )

# Preprocessing categorical inputs to match standard label encodings
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0

region_mapping = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
region_val = region_mapping[region]

st.markdown("")

# Prediction trigger
if st.button("Predict Cost"):
    # Features array mapped in the required training order:
    # [age, sex, bmi, children, smoker, region]
    features = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
    prediction = model.predict(features)[0]

    st.markdown(
        f"""
        <div class="prediction-card">
            Estimated Insurance Cost: ${prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )
