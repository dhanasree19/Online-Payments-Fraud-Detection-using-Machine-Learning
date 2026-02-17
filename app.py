import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Page config
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="💳",
    layout="centered"
)

# Title
st.markdown("<h1 style='text-align: center;'>💳 Online Payment Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "About"])


# ================= HOME PAGE =================
if page == "Home":

    st.subheader("🏠 Welcome")

    st.write("""
    This application predicts whether an online transaction is **Fraudulent or Safe**
    using Machine Learning.
    
    👉 Go to **Predict Page** to check your transaction.
    """)

    st.success("Deployed using Streamlit Cloud 🚀")


# ================= PREDICT PAGE =================
elif page == "Predict":

    st.subheader("🔍 Enter Transaction Details")

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
        isFlaggedFraud = st.number_input("Is Flagged Fraud (0 or 1)", min_value=0.0, max_value=1.0)

    st.markdown("")

    if st.button("🔎 Predict"):

        data = np.array([[step, type_t, amount,
                          oldOrg, newOrg,
                          oldDest, newDest,
                          isFlaggedFraud]])

        prediction = model.predict(data)[0]

        st.markdown("<hr>", unsafe_allow_html=True)

        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Safe Transaction")


# ================= ABOUT PAGE =================
elif page == "About":

    st.subheader("📘 About Project")

    st.write("""
    **Project Name:** Online Payment Fraud Detection  
    
    **Technologies Used:**
    - Python
    - Scikit-learn
    - Streamlit
    
    **Purpose:**
    To detect fraudulent online transactions using Machine Learning.
    """)

    st.info("Created for Academic Project & Demo")
