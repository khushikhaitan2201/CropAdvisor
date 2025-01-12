import streamlit as st
import numpy as np
import pickle

# Load the model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
    15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# App title
st.title("🌱 Crop Recommendation System")

# Input form
st.sidebar.header("Enter Crop Data")
with st.sidebar.form(key='crop_form'):
    N = st.number_input("Nitrogen Content (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
    P = st.number_input("Phosphorus Content (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
    K = st.number_input("Potassium Content (K)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    submit_button = st.form_submit_button(label="Predict Crop")

# Prediction
if submit_button:
    try:
        # Prepare the input features
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply scalers
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Predict crop
        prediction = model.predict(final_features)[0]

        # Display the result
        if prediction in crop_dict:
            crop = crop_dict[prediction]
            st.success(f"🌾 Recommended Crop: **{crop}** is the best crop to be cultivated.")
        else:
            st.error("Unable to determine the best crop with the provided data.")
    except Exception as e:
        st.error(f"Error: {e}")