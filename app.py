import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Set the page layout to wide
st.set_page_config(layout="wide")

# Function to load the model checkpoint; assumes model was saved with pickle
@st.cache_resource
def load_model(filepath):
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model

# Load the model from the checkpoint file (update the path if needed)
model = load_model("synapses_checkpint.pkt")

# Define two columns: left for inputs, right for outputs
col1, col2 = st.columns([1, 2])

# Generate 12 input features
features = [f"f{i}" for i in range(12)]
feature_values = {}

with col1:
    st.header("Input Features")
    # Create a slider for each feature
    for feat in features:
        feature_values[feat] = st.slider(feat, 0, 100, 50)

# Create a DataFrame from the feature values, used later for plotting
df = pd.DataFrame(list(feature_values.items()), columns=["Feature", "Value"])

with col2:
    st.header("Predicted Conversion Rate")
    
    # Prepare the features for model input:
    # The model expects an array of shape (n_samples, n_features)
    input_features = [list(feature_values.values())]
    
    # Perform inference using the loaded model
    try:
        prediction = model.predict(input_features)
        # We assume the model returns a list or array, with the first (only) element as the conversion rate
        conversion_rate = prediction[0]
    except Exception as e:
        conversion_rate = None
        st.error(f"An error occurred during inference: {e}")
    
    # Display the conversion rate if inference succeeded
    if conversion_rate is not None:
        st.metric(label="Estimated Conversion Rate", value=f"{conversion_rate:.2f}%")
    
    st.header("Feature Importance")
    # Check if the model has feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        # Create a bar chart for feature importances
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df["Feature"], feature_importances, color='skyblue')
        ax.set_title("Feature Importance")
        ax.set_ylabel("Importance")
        plt.xticks(rotation=45, ha="right")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
    else:
        st.warning("Model does not support feature importance visualization.")

    # Optionally, display the feature values chart as a reference
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["Feature"], df["Value"], color='skyblue')
    ax.set_title("Feature Values")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
