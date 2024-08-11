import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('road_prediction_model.pkl')

# Title of the app
st.title('Road Construction Prediction App')

# Input fields
length_of_road = st.number_input('Length of the Road')
breadth_of_road = st.number_input('Breadth of the Road')
structural_number = st.number_input('Structural Number')
duration_of_project = st.number_input('Duration of the Project')

# Prediction button
if st.button('Predict'):
    # Prepare input for prediction
    input_data = np.array([[length_of_road, breadth_of_road, structural_number, duration_of_project]])
    
    # Make predictions
    prediction = model.predict(input_data)
    
    # Calculate budget
    thickness_of_road = prediction[0][0]
    estimation_budget = length_of_road * breadth_of_road * thickness_of_road * 12000
    
    # Display the predictions
    st.subheader('Predicted Outputs:')
    st.write(f"Thickness of Road: {prediction[0][0]}")
    st.write(f"Number of Paving Machines: {prediction[0][1]}")
    st.write(f"Number of Compactors: {prediction[0][2]}")
    st.write(f"Number of Working Men: {prediction[0][3]}")
    st.write(f"Ratio of Aggregate: {prediction[0][4]}")
    st.write(f"Temperature: {prediction[0][5]}")
    st.write(f"Estimation Budget: {estimation_budget}")  # Calculated budget