import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

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

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print(f'Cross-Validation RMSE: {cv_rmse.mean():.2f}')

# Predict on the test set
y_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Test RMSE: {test_rmse:.2f}')

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

future_prediction = model.predict(future_data)
print(f'Predicted "Strongly Agree" percentage for 2020: {future_prediction[0]:.2f}%')
