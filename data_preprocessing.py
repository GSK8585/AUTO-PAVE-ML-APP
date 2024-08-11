import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('road_construction_data - Sheet1.csv')

# Preprocessing steps
X = data[['LENGTH OF THE ROAD', 'BREADTH OF ROAD', 'STRUCTURAL NUMBER', 'DURATION OF THE PROJECT']]
y = data[['Thickness of Road', 'No of Paving Machines', 'No of Compactors', 
          'No of Working Men', 'Ratio of Aggregate', 'TEMPERATURE']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)