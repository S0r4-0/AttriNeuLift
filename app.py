import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Set the page layout and title
st.set_page_config(page_title="Attri-NeuLift", layout="wide")

st.title("Attri-NeuLift")

# Load the model checkpoint
@st.cache_resource
def load_model(filepath):
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model("synapses_checkpint.pkt")

# Define columns
col1, col2 = st.columns([1, 2])

# Define input features
features = [f"f{i}" for i in range(12)]
feature_values = {}

with col1:
    st.header("Features values")
    # Sliders with 0.01 step for all
    feature_values['f0'] = st.slider('f0', 12.0, 27.0, 20.0, step=0.01)
    feature_values['f1'] = st.slider('f1', 10.0, 17.0, 14.0, step=0.01)
    feature_values['f2'] = st.slider('f2', 8.0 ,9.5, 8.6, step=0.01)
    feature_values['f3'] = st.slider('f3', -8.4, 4.7, 0.0, step=0.01)
    feature_values['f4'] = st.slider('f4', 10.0, 21.1, 15.0, step=0.01)
    feature_values['f5'] = st.slider('f5', -9.0, 4.2, -2.0, step=0.01)
    feature_values['f6'] = st.slider('f6', -31.0, 0.3, -15.0, step=0.01)
    feature_values['f7'] = st.slider('f7', 4.83, 7.0, 6.0, step=0.01)
    feature_values['f8'] = st.slider('f8', 3.64, 4.0, 3.8, step=0.01)
    feature_values['f9'] = st.slider('f9', 13.0, 75.0, 40.0, step=0.01)
    feature_values['f10'] = st.slider('f10', 5.0, 6.5, 5.75, step=0.01)
    feature_values['f11'] = st.slider('f11', -1.4, -0.1, -0.8, step=0.01)

# DataFrame for visualization
df = pd.DataFrame(list(feature_values.items()), columns=["Feature", "Value"])

with col2:
    st.header("Predicted Conversion Rate")

    # Model input
    input_features = np.array([list(feature_values.values())])

    # Predict
    try:
        prediction = model.predict(input_features)
        conversion_rate = prediction[0]
    except Exception as e:
        conversion_rate = None
        st.error(f"An error occurred during inference: {e}")

    # Display prediction
    if conversion_rate is not None:
        st.metric(label="Estimated Conversion Rate", value=f"{conversion_rate.item() * 100:.2f}%")

    # Feature bar chart
    st.header("Feature Values")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["Feature"], df["Value"], color='skyblue')
    ax.set_title("Feature Values")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
