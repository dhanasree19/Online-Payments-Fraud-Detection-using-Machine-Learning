import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")


# Page Config
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# Session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ================== HOME PAGE ==================
if st.session_state.page == "Home":

    st.markdown("<h1 style='text-align:center;'>💳 Online Payment Fraud Detection</h1>",
                unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.write("""
    Welcome to the Fraud Detection System.

    This application predicts whether a transaction is:

    ✅ Safe  
    ⚠️ Fraudulent
    """)

    st.info("Click below to start prediction")

    if st.button("🚀 Go to Predict"):
        st.session_state.page = "Predict"
        st.experimental_rerun()


# ================== PREDICT PAGE ==================
elif st.session_state.page == "Predict":

    st.markdown("<h2 style='text-align:center;'>🔍 Enter Transaction Details</h2>",
                unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input("Step", min_value=0.0)
        type_t = st.number_input("Transaction Type", min_value=0.0)
        amount = st.number_input("Amount", min_value=0.0)
        oldOrg = st.number_input("Old Balance (Sender)", min_value=0.0)

    with col2:
        newOrg = st.number_input("New Balance (Sender)", min_value=0.0)
        oldDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
        newDest = st.number_input("New Balance (Receiver)", min_value=0.0)
        isFlagged = st.number_input("Is Flagged Fraud (0 or 1)", min_value=0.0, max_value=1.0)

    st.markdown("")

    if st.button("🔎 Predict"):

        data = np.array([[step, type_t, amount,
                          oldOrg, newOrg,
                          oldDest, newDest,
                          isFlagged]])

        pred = model.predict(data)[0]

        st.session_state.prediction = pred
        st.session_state.page = "Result"

        st.experimental_rerun()

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"
        st.experimental_rerun()


# ================== RESULT PAGE ==================
elif st.session_state.page == "Result":

    st.markdown("<h2 style='text-align:center;'>📊 Prediction Result</h2>",
                unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state.prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Safe Transaction")

    st.markdown("")

    if st.button("🔁 Predict Again"):
        st.session_state.page = "Predict"
        st.experimental_rerun()

    if st.button("🏠 Back to Home"):
        st.session_state.page = "Home"
        st.experimental_rerun()
