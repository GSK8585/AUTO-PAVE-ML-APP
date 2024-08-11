import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
df = pd.read_csv('road_construction_data - Sheet1.csv')

# Function to extract numeric ratio of aggregate
def extract_aggregate_ratio(ratio_str):
    # Extract the percentage part of the string
    aggregate_ratio = float(ratio_str.split('%')[0])
    return aggregate_ratio

# Apply this function to the "Ratio of Aggregate" column
df['Ratio of Aggregate'] = df['Ratio of Aggregate'].apply(extract_aggregate_ratio)

# Define your features (X) and target (y)
X = df[['Length of Road', 'Breadth of Road', 'Structural Number', 'Duration of Project']]
y = df[['Thickness of Road', 'Number of Paving Machines', 'Number of Compactors', 'Number of Working Men', 'Ratio of Aggregate', 'Temperature']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed data for use in model_training.py
joblib.dump((X_train, X_test, y_train, y_test), 'preprocessed_data.pkl')