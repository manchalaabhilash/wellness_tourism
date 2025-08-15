import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# HF Model Repo
HF_MODEL_REPO = "abhilashmanchala/wellness_tourism_model"

# If model is private, use token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Download model
model_path = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename="best_tourism_model_v1.joblib",
    repo_type="model",
    token=HF_TOKEN
)

preprocessor_path = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename="preprocessor.joblib",
    repo_type="model",
    token=HF_TOKEN
)

# Load model and preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

st.title("🏝 Wellness Tourism Package Prediction")
st.write("Fill the details below to predict if the customer will purchase the package.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, step=1)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Other"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons = st.number_input("Number Of Persons Visiting", min_value=1, step=1)
preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number Of Trips", min_value=0, step=1)
passport = st.selectbox("Has Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children = st.number_input("Number Of Children Visiting", min_value=0, step=1)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "Other"])
monthly_income = st.number_input("Monthly Income", min_value=0, step=1000)
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "Other"])
num_followups = st.number_input("Number Of Followups", min_value=0, step=1)
pitch_duration = st.number_input("Duration Of Pitch", min_value=0, step=1)

# Create DataFrame for input
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": pitch_duration
}])

if st.button("Predict"):
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)[0]

    if prediction == 1:
        st.success("✅ Customer is likely to purchase the Wellness Tourism Package!")
    else:
        st.error("❌ Customer is not likely to purchase the package.")
