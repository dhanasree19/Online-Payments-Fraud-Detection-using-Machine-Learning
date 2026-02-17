import streamlit as st
import numpy as np
import joblib


# Load Model
model = joblib.load("random_forest_model.pkl")


# Page Setup
st.set_page_config(
    page_title="Online Payment Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# Session State
if "page" not in st.session_state:
    st.session_state.page = "home"

if "result" not in st.session_state:
    st.session_state.result = ""


# ---------------- HOME PAGE ----------------
def home():

    st.title("💳 Online Payment Fraud Detection")
    st.subheader("Machine Learning Based System")

    st.write("---")

    st.write("""
    This application predicts whether an online transaction 
    is **Fraudulent** or **Safe** using Machine Learning.
    """)

    st.write("### 📌 Features")
    st.markdown("""
    - Secure Prediction  
    - Fast Processing  
    - Easy Interface  
    - Public Access  
    """)

    if st.button("🚀 Start Prediction"):
        st.session_state.page = "predict"
        st.rerun()


# ---------------- PREDICT PAGE ----------------
def predict():

    st.title("📝 Enter Transaction Details")

    st.write("---")

    with st.form("prediction_form"):

        step = st.number_input("Step", min_value=0.0)
        type_t = st.number_input("Transaction Type (Encoded)", min_value=0.0)
        amount = st.number_input("Amount", min_value=0.0)

        oldOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
        newOrg = st.number_input("New Balance (Sender)", min_value=0.0)

        oldDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
        newDest = st.number_input("New Balance (Receiver)", min_value=0.0)

        isFlaggedFraud = st.number_input("Is Flagged Fraud (0 or 1)", min_value=0.0, max_value=1.0)

        submit = st.form_submit_button("🔍 Predict")


    if submit:

        data = np.array([[

            step,
            type_t,
            amount,
            oldOrg,
            newOrg,
            oldDest,
            newDest,
            isFlaggedFraud

        ]])

        prediction = model.predict(data)[0]

        if prediction == 1:
            st.session_state.result = "⚠️ Fraud Transaction Detected!"
        else:
            st.session_state.result = "✅ Safe Transaction"

        st.session_state.page = "result"
        st.rerun()


    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()


# ---------------- RESULT PAGE ----------------
def result():

    st.title("📊 Prediction Result")

    st.write("---")

    if "Fraud" in st.session_state.result:
        st.error(st.session_state.result)
    else:
        st.success(st.session_state.result)


    st.write("### 🔁 What would you like to do next?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 New Prediction"):
            st.session_state.page = "predict"
            st.rerun()

    with col2:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()



# ---------------- ROUTING ----------------
if st.session_state.page == "home":
    home()

elif st.session_state.page == "predict":
    predict()

elif st.session_state.page == "result":
    result()
