import streamlit as st
import pandas as pd
import joblib

from train import MODEL_PATH, train_model


@st.cache_resource
def load_model():
    """Load the saved model, training it first if it does not exist yet."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        with st.spinner("First run: training the model (takes a few seconds)..."):
            model, _ = train_model()
        return model


model = load_model()

st.title("🏡 California House Price Predictor")
st.write("This app predicts the **Median House Value** in California based on various features. It uses a Random Forest Regressor trained on the California Housing dataset.")

st.sidebar.header("Input Features")

def user_input_features():
    MedInc = st.sidebar.slider("Median Income (in $10,000)", 0.0, 15.0, 3.0)
    HouseAge = st.sidebar.slider("House Age (Years)", 1, 52, 20)
    AveRooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
    AveBedrms = st.sidebar.slider("Average Bedrooms", 1.0, 5.0, 1.0)
    Population = st.sidebar.slider("Population", 10, 5000, 1000)
    AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
    Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 35.0)
    Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -119.0)
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Selected Features")
st.write(input_df)

if st.button("Predict House Value"):
    prediction = model.predict(input_df)
    st.success(f"### Predicted Median House Value: **${prediction[0]*100000:,.2f}**")
