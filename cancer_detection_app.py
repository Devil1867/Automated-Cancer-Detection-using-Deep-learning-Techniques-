
# Cancer Detection Web App using Streamlit

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model (make sure the model file exists in the same directory)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cnn_model.h5")
    return model

model = load_model()

# Page Title
st.title("🧠 Cancer Detection AI")
st.markdown("Upload patient feature data to predict **Benign** or **Malignant** cancer likelihood.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with patient features (no ID/Diagnosis column)", type=["csv"])

if uploaded_file is not None:
    try:
        # Read and display the input data
        input_df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.dataframe(input_df)

        # Preprocess: Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(input_df)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Predict
        predictions = model.predict(X_reshaped)

        st.subheader("🔍 Prediction Results")
        for i, pred in enumerate(predictions):
            benign = pred[0] * 100
            malignant = pred[1] * 100
            diagnosis = "🔴 Malignant" if malignant > benign else "🟢 Benign"

            st.markdown(f"""
**Patient {i+1}:**
- Benign Probability: `{benign:.2f}%`
- Malignant Probability: `{malignant:.2f}%`
- **Prediction**: {diagnosis}
""")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file with patient features to begin.")
