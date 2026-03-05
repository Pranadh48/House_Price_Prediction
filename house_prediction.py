# House Price Prediction
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Load the dataset
data = pd.read_csv('house_price_500_records.csv')
# Preprocess the data
data = data.dropna()  # Remove missing values

# Define features and target variable
X = data[['square_feet']]  # Features
y = data[['price']]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# streamlit application
import streamlit as st
st.title("House Price Prediction")
st.write("This application predicts house prices based on various features.")

# User input for features
sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=1500)

# Create a DataFrame for the user input features
input_data = pd.DataFrame({
    'square_feet': [sqft]
})
# Predict the house price based on user input
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)
    st.write(f"Predicted House Price: ${predicted_price[0][0]:.2f}")
    
#visualization
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, y_pred, color='red', label='Predicted Prices')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
st.pyplot(plt)

    