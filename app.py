import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🛡️ Multi-Phase Fraud Detection System")
st.markdown("---")

# --- LOAD MODELS ---
@st.cache_resource
def get_models():
    rf = joblib.load('models/rf_baseline_model.joblib')
    xgb = joblib.load('models/xgboost_smote_model.joblib')
    # Custom object fix for the MSE error
    auto = load_model('models/autoencoder_fraud_model.h5', 
                      custom_objects={'mse': MeanSquaredError()})
    return rf, xgb, auto

rf_model, xgb_model, autoencoder = get_models()

# --- SIDEBAR ---
st.sidebar.header("🕹️ Control Panel")
if st.sidebar.button("Generate Random Transaction"):
    # Generate 30 features (matches RF/XGB training)
    st.session_state['sample'] = np.random.randn(1, 30)

# --- MAIN UI ---
if 'sample' in st.session_state:
    # 1. TOP SECTION: DATA INFORMATION
    st.subheader("📊 Transaction Feature Analysis")
    # Show the features as a clean table
    cols_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    df_display = pd.DataFrame(st.session_state['sample'], columns=cols_names)
    st.dataframe(df_display.style.highlight_max(axis=1))

    # 2. MIDDLE SECTION: THE 3-PHASE DETECTION
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Phase 1: Baseline")
        pred1 = rf_model.predict(st.session_state['sample'])
        status1 = "🔴 FRAUD" if pred1 == 1 else "🟢 NORMAL"
        st.subheader(status1)
        st.caption("Standard Random Forest Check")

    with col2:
        st.header("Phase 2: AML")
        # 1. Get the probabilities (this returns a list of lists)
        probabilities = xgb_model.predict_proba(st.session_state['sample'])
        
        # 2. Extract the 'Fraud' probability (the 2nd value in the 1st row)
        # .flatten() ensures it's a simple list, then [1] gets the Fraud class
        prob2 = float(probabilities.flatten()[1])
        
        # 3. Now the formatting will work perfectly
        st.metric("Risk Score", f"{prob2 * 100:.1f}%")
        
        status2 = "🔴 HIGH RISK" if prob2 > 0.5 else "🟢 SAFE"
        st.write(f"Decision: **{status2}**")

    with col3:
        st.header("Phase 3: Deep Learning")
        # Autoencoder only uses first 29 features
        sample_29 = st.session_state['sample'][:, :29]
        recon = autoencoder.predict(sample_29)
        mse = np.mean(np.power(sample_29 - recon, 2))
        
        st.metric("Reconstruction Error (MSE)", f"{mse:.4f}")
        status3 = "⚠️ ANOMALY" if mse > 0.7 else "✅ AUTHENTIC"
        st.write(f"Integrity: **{status3}**")

    # 3. BOTTOM SECTION: CONCLUSION
    st.info("**Information Summary:** Traditional models focus on known patterns (Phase 1 & 2). Phase 3 uses an Autoencoder to detect unknown 'Anomalies'—transactions that don't fit the historical mathematical profile.")
else:
    st.warning("Please click the 'Generate Random Transaction' button in the sidebar to begin analysis.")