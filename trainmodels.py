
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Preprocess the dataset
# Encode categorical variables
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])
data['Occupation'] = encoder.fit_transform(data['Occupation'])
data['BMI Category'] = encoder.fit_transform(data['BMI Category'])

# Feature selection
X = data.drop(['Person ID', 'Blood Pressure', 'Sleep Duration'], axis=1)
y = data['Sleep Duration']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save the model
joblib.dump(model, "decision_tree_regressor.pkl")
print("Model saved as decision_tree_regressor.pkl")
