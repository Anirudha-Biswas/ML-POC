import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('BP_Predictor/bp_model_v1.pkl')

# App UI
st.title("🩺 Blood Pressure Predictor")
st.markdown("Enter your details to predict your blood pressure.")

# User input
age = st.slider("Age", 18, 90, 30)
weight = st.slider("Weight (kg)", 40, 120, 70)
lifestyle = st.selectbox("Lifestyle", ['active', 'sedentary', 'smoker'])

if st.button("Predict Blood Pressure"):
    input_data = pd.DataFrame({
        'Age': [age],
        'Weight': [weight],
        'Lifestyle': [lifestyle]
    })
    
    # Predict with all decision trees to get confidence interval
    regressor = model.named_steps['regressor']
    preprocessed_input = model.named_steps['preprocessor'].transform(input_data)
    all_predictions = np.array([tree.predict(preprocessed_input)[0] for tree in regressor.estimators_])
    
    predicted_mean = np.mean(all_predictions)
    predicted_std = np.std(all_predictions)
    lower = predicted_mean - 1.96 * predicted_std
    upper = predicted_mean + 1.96 * predicted_std

    st.success(f"Predicted Blood Pressure: {predicted_mean:.2f} mmHg")
    st.info(f"95% Confidence Interval: {lower:.2f} – {upper:.2f} mmHg")

    # Plotting predictions from all trees
    st.subheader("Prediction Spread from All Trees")
    fig, ax = plt.subplots()
    ax.hist(all_predictions, bins=10, color='skyblue', edgecolor='black')
    ax.axvline(predicted_mean, color='red', linestyle='dashed', linewidth=2, label='Mean Prediction')
    ax.axvline(lower, color='green', linestyle='dotted', label='Lower Bound (95%)')
    ax.axvline(upper, color='green', linestyle='dotted', label='Upper Bound (95%)')
    ax.set_xlabel("Blood Pressure Predictions")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
