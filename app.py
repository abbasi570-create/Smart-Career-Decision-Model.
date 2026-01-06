import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

st.set_page_config(page_title="Career Predictor", page_icon="💼", layout="centered")
st.title("💼 Career Prediction App")
st.write("Select your skills, personality, and interests to find your ideal career!")

# --------------------------
# 1️⃣ Dataset
# --------------------------
data = [
    {
        "Computer_Skill": 5,
        "Confidence": 4,
        "Environment": "Office",
        "Creativity": 5,
        "Social": "Moderate",
        "Marks": 5,
        "Interests": ["Coding", "Photography", "Writing"],
        "Fear_of_Blood": "No",
        "Allergies": "None",
        "Leadership": 4,
        "Physical_Fitness": 3,
        "Risk_Taking": 3,
        "Patience": 4,
        "Analytical_Thinking": 5,
        "Night_Shift_Tolerance": "Yes",
        "Communication_Skills": 4,
        "Career": "Software Developer"
    },
    {
        "Computer_Skill": 2,
        "Confidence": 3,
        "Environment": "Home",
        "Creativity": 4,
        "Social": "Introvert",
        "Marks": 4,
        "Interests": ["Writing", "Helping People", "Photography"],
        "Fear_of_Blood": "No",
        "Allergies": "Mild",
        "Leadership": 3,
        "Physical_Fitness": 3,
        "Risk_Taking": 2,
        "Patience": 5,
        "Analytical_Thinking": 4,
        "Night_Shift_Tolerance": "No",
        "Communication_Skills": 4,
        "Career": "Content Creator / Freelance Writer"
    },
    # ------------------ add your remaining dataset items here ------------------
]

# --------------------------
# 2️⃣ Convert to DataFrame
# --------------------------
df = pd.DataFrame(data)

# --------------------------
# 3️⃣ Train or load model
# --------------------------
try:
    # Try to load existing model & encoders
    model = joblib.load("career_model.pkl")
    mlb = joblib.load("interests_encoder.pkl")
    le = joblib.load("career_label_encoder.pkl")
except:
    st.info("Training model for the first time... please wait ⏳")
    
    # Encode Interests (multi-label)
    mlb = MultiLabelBinarizer()
    interests_encoded = mlb.fit_transform(df['Interests'])

    # Encode categorical features
    df['Environment'] = LabelEncoder().fit_transform(df['Environment'])
    df['Social'] = LabelEncoder().fit_transform(df['Social'])
    df['Fear_of_Blood'] = LabelEncoder().fit_transform(df['Fear_of_Blood'])
    df['Allergies'] = LabelEncoder().fit_transform(df['Allergies'])
    df['Night_Shift_Tolerance'] = LabelEncoder().fit_transform(df['Night_Shift_Tolerance'])

    # Combine numeric + interests
    X_numeric = df[['Computer_Skill','Confidence','Environment','Creativity','Social','Marks',
                    'Fear_of_Blood','Allergies','Leadership','Physical_Fitness','Risk_Taking',
                    'Patience','Analytical_Thinking','Night_Shift_Tolerance','Communication_Skills']].values
    X = np.hstack([X_numeric, interests_encoded])

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['Career'])

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Save model & encoders
    joblib.dump(model, "career_model.pkl")
    joblib.dump(mlb, "interests_encoder.pkl")
    joblib.dump(le, "career_label_encoder.pkl")
    st.success("✅ Model trained and saved!")

# --------------------------
# 4️⃣ Streamlit Inputs
# --------------------------
st.header("🔹 Your Attributes")

# Numeric dropdowns
def number_selector(label, min_val, max_val, default):
    return st.selectbox(label, list(range(min_val, max_val+1)), index=default-min_val)

Computer_Skill = number_selector("Computer Skill", 1, 5, 3)
Confidence = number_selector("Confidence", 1, 5, 3)
Creativity = number_selector("Creativity", 1, 5, 3)
Marks = number_selector("Marks / Scores", 1, 5, 3)
Leadership = number_selector("Leadership", 1, 5, 3)
Physical_Fitness = number_selector("Physical Fitness", 1, 5, 3)
Risk_Taking = number_selector("Risk Taking", 1, 5, 3)
Patience = number_selector("Patience", 1, 5, 3)
Analytical_Thinking = number_selector("Analytical Thinking", 1, 5, 3)
Communication_Skills = number_selector("Communication Skills", 1, 5, 3)

# Categorical dropdowns
Environment = st.selectbox("Preferred Work Environment", ["Office", "Home", "Field", "Flexible"])
Social = st.selectbox("Social Type", ["Introvert", "Moderate", "Extrovert"])
Fear_of_Blood = st.selectbox("Fear of Blood?", ["Yes", "No"])
Allergies = st.selectbox("Allergies", ["None", "Mild", "Severe"])
Night_Shift_Tolerance = st.selectbox("Can work Night Shifts?", ["Yes", "No"])

# Interests multi-select
all_interests = ["Coding", "Photography", "Writing", "Helping People", "Public Speaking",
                 "Event Planning", "Driving", "Exploring", "Graphic Design", "Teaching",
                 "Marketing", "Finance", "Strategy", "Leadership", "Healthcare", "Acting"]
Interests = st.multiselect("Your Interests", all_interests)

# --------------------------
# 5️⃣ Prediction
# --------------------------
st.markdown("---")
if st.button("Predict Career"):
    # Encode categorical values
    env_val = ["Office", "Home", "Field", "Flexible"].index(Environment)
    social_val = ["Introvert", "Moderate", "Extrovert"].index(Social)
    blood_val = ["No", "Yes"].index(Fear_of_Blood)
    allergy_val = ["None", "Mild", "Severe"].index(Allergies)
    night_val = ["No", "Yes"].index(Night_Shift_Tolerance)
    
    # Encode interests
    interests_array = mlb.transform([Interests])
    
    # Combine all inputs
    X_input = np.array([[Computer_Skill, Confidence, env_val, Creativity, social_val, Marks,
                         blood_val, allergy_val, Leadership, Physical_Fitness, Risk_Taking,
                         Patience, Analytical_Thinking, night_val, Communication_Skills]])
    
    X_final = np.hstack([X_input, interests_array])
    
    # Predict career
    pred = model.predict(X_final)
    career = le.inverse_transform(pred)
    
    # Display result
    st.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#D4EDDA; color:#155724; font-size:20px;'>🎯 Predicted Career: <b>{career[0]}</b></div>", unsafe_allow_html=True)
