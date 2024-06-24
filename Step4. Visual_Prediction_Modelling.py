import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/paritagala/Desktop/Website/2. Employee Engagement and Productivity Analysis/employee_engagement_data.csv'
data = pd.read_csv(file_path)

# Exclude non-numeric columns
numeric_columns = ['Strongly Agree', 'Agree', 'Neutral', 'Disagree', 'Strongly Disagree']
data_numeric = data[numeric_columns + ['Year']]

# Feature engineering: Create a feature matrix X and target vector y
X = data_numeric.drop(columns=['Strongly Agree'])
y = data_numeric['Strongly Agree']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection: Using Linear Regression
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Making future predictions
# Assuming we want to predict for the year 2020 with average values for Agree, Neutral, Disagree, Strongly Disagree
future_data = pd.DataFrame({
    'Year': [2020],
    'Agree': [data['Agree'].mean()],
    'Neutral': [data['Neutral'].mean()],
    'Disagree': [data['Disagree'].mean()],
    'Strongly Disagree': [data['Strongly Disagree'].mean()]
})

# Ensure the order of columns in future_data matches the order used in training
future_data = future_data[X_train.columns]

# Predict the future value
future_prediction = model.predict(future_data)[0]

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Test Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual "Strongly Agree" %')
plt.ylabel('Predicted "Strongly Agree" %')
plt.title('Actual vs Predicted "Strongly Agree" %')
plt.legend()
plt.show()

# Adding future prediction to the plot
years = data['Year'].unique()
mean_strongly_agree_by_year = data.groupby('Year')['Strongly Agree'].mean()

plt.figure(figsize=(10, 6))
plt.plot(years, mean_strongly_agree_by_year, marker='o', linestyle='-', label='Actual Mean "Strongly Agree" %')
plt.scatter([2020], [future_prediction], color='red', label='Predicted "Strongly Agree" % for 2020')
plt.xlabel('Year')
plt.ylabel('"Strongly Agree" %')
plt.title('Mean "Strongly Agree" % by Year with Prediction for 2020')
plt.legend()
plt.show()
