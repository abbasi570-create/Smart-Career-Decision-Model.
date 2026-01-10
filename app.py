import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Smart Career Predictor", page_icon="🚀", layout="wide")

# -------------------------
# Load Model
# -------------------------
model = joblib.load("career_model.pkl")
le = joblib.load("career_label.pkl")

# -------------------------
# Header
# -------------------------
st.markdown("""
<style>
.header {
    background: linear-gradient(90deg,#ff7e5f,#feb47b);
    padding: 15px;
    border-radius: 10px;
    text-align:center;
}
</style>
<div class="header"><h1>🚀 Smart Career Prediction Based on Skills</h1></div>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Enter Your Scores")

def user_input():
    return pd.DataFrame([{
        "Math": st.sidebar.slider("Math", 0, 100, 70),
        "Science": st.sidebar.slider("Science", 0, 100, 70),
        "Biology": st.sidebar.slider("Biology", 0, 100, 70),
        "Tech_Interest": st.sidebar.slider("Interest in Technology", 0, 10, 5),
        "Med_Interest": st.sidebar.slider("Interest in Medicine", 0, 10, 5),
        "Business_Interest": st.sidebar.slider("Interest in Business", 0, 10, 5),
        "Creativity": st.sidebar.slider("Creativity", 0, 10, 5),
        "Logical": st.sidebar.slider("Logical Thinking", 0, 10, 5)
    }])

input_df = user_input()

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Career 🚀"):
    pred = model.predict(input_df)
    career = le.inverse_transform(pred)[0]

    st.success(f"🎯 Recommended Career: **{career}**")
    st.balloons()

    # Probability chart
    probs = model.predict_proba(input_df)[0]
    prob_df = pd.DataFrame({
        "Career": le.classes_,
        "Probability": probs
    })

    fig = px.bar(prob_df, x="Career", y="Probability", color="Career", text="Probability")
    st.plotly_chart(fig)
