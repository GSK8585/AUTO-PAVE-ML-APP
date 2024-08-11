import streamlit as st
import joblib
import numpy as np

# Load the trained model
# Make sure 'road_prediction_model.pkl' is in the same directory as this script
model = joblib.load('road_prediction_model.pkl')

# Title of the app
st.title('Road Construction Prediction App')

# Input fields for the user to provide the necessary information
length_of_road = st.number_input('Length of the Road (meters)', min_value=0.0)
breadth_of_road = st.number_input('Breadth of the Road (meters)', min_value=0.0)
structural_number = st.number_input('Structural Number', min_value=0.0)
duration_of_project = st.number_input('Duration of the Project (days)', min_value=0.0)

# Button to make predictions
if st.button('Predict'):
    # Check if all inputs are provided
    if length_of_road > 0 and breadth_of_road > 0 and structural_number > 0 and duration_of_project > 0:
        # Prepare input data for the model prediction
        input_data = np.array([[length_of_road, breadth_of_road, structural_number, duration_of_project]])

        # Make predictions
        prediction = model.predict(input_data)
        
        # Extract the predictions
        thickness_of_road = prediction[0][0]
        num_paving_machines = prediction[0][1]
        num_compactors = prediction[0][2]
        num_working_men = prediction[0][3]
        ratio_of_aggregate = prediction[0][4]
        temperature = prediction[0][5]

        # Calculate the estimation budget based on the provided formula
        estimation_budget = length_of_road * breadth_of_road * thickness_of_road * 12000

        # Display the results to the user
        st.subheader('Predicted Outputs:')
        st.write(f"Thickness of Road: {thickness_of_road:.2f} meters")
        st.write(f"Number of Paving Machines: {num_paving_machines:.0f}")
        st.write(f"Number of Compactors: {num_compactors:.0f}")
        st.write(f"Number of Working Men: {num_working_men:.0f}")
        st.write(f"Ratio of Aggregate: {ratio_of_aggregate:.2f}")
        st.write(f"Temperature: {temperature:.2f} °C")
        st.write(f"Estimation Budget: ${estimation_budget:,.2f}")
    else:
        st.error('Please fill in all the input fields with valid values.')