import pandas as pd

# Load the dataset
file_path = '/Users/paritagala/Desktop/Website/2. Employee Engagement and Productivity Analysis/employee_engagement_data.csv'
data = pd.read_csv(file_path)

# Exclude non-numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
overall_engagement = data.groupby('Year')[numeric_columns].mean()

# Display the results
print(overall_engagement)

# Find the questions with the highest and lowest "Strongly Agree" responses for each year
highest_engagement = data.loc[data.groupby('Year')['Strongly Agree'].idxmax()]
lowest_engagement = data.loc[data.groupby('Year')['Strongly Agree'].idxmin()]

highest_engagement[['Year', 'Survey Question Number', 'Survey Question Text', 'Strongly Agree']]
