import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import h5py

model = load_model(r"C:\Users\DELL\my_adhd_model.h5")  # Update with the correct path

# Preprocess data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Predict ADHD
def predict_adhd(mat_file):
    with h5py.File(mat_file, 'r') as file:
        # Assuming your data is stored in a dataset named 'data'
        data_array = file['data'][:]
    
    # Convert to DataFrame
    df = pd.DataFrame(data_array)

    # Ensure DataFrame is not empty
    if not df.empty:
        # Preprocess the data and make prediction
        processed_data = preprocess_data(df)
        prediction = model.predict(processed_data)

        if prediction is not None and len(prediction) > 0:
            # Take the mean of the prediction array
            mean_prediction = prediction.mean()
            st.write(mean_prediction)

            # Check prediction and return result
            if mean_prediction < 0.28:
                return "Non Diagnostic"
            elif 0.28 <= mean_prediction < 0.45:
                return "Dyslexia"
            elif 0.45 <= mean_prediction < 0.65:
                return "ADHD"
            else:
                return "Dyslexia and ADHD"
        else:
            return "Failed to make prediction. Please ensure the data format is correct."
    else:
        return "Failed to create DataFrame from the data."

# Main function to run the Streamlit app
def main():
    st.title("ADHD and Dyslexia Prediction App")
    st.write("Upload .mat file to predict ADHD and Dyslexia")

    uploaded_file = st.file_uploader("Choose a .mat file", type="mat")

    if uploaded_file is not None:
        # Predict ADHD
        prediction = predict_adhd(uploaded_file)

        # Display prediction result
        st.write("Prediction:", prediction)

# Run the main function
if __name__ == "__main__":
    main()
