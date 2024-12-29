import pandas as pd

# Load the DataSet:
data = pd.read_csv('F:/Data Science/Intro to Data Science/stroke-dataset-analysis/stroke-data.csv')
print(data.head())
print()

'''
PART #1:
Exploratory Data Analysis (EDA):
'''

''' 1. Summary Statistics: '''
print('Summary Statistics:\n', data.describe())

''' 2. Visualizations: '''
import matplotlib.pyplot as plt
import seaborn as sns

# Count plot for stroke:
plt.figure(figsize=(8, 6))
sns.countplot(x='stroke', data=data)
plt.title('Count of Strokes')
plt.xlabel('Stroke (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

''' 3. Correlation Analysis: '''
# Convert categorical variables to numeric (if needed)
data_encoded = pd.get_dummies(data, drop_first=True)  # Convert all categorical variables to numeric

# Select only numeric columns for correlation
data_numeric = data_encoded.select_dtypes(include=['number'])

correlation = data_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

''' 4. Missing Value Analysis: '''
missing_values = data.isnull().sum()
print("\nMissing Values:\n", missing_values[missing_values > 0])

''' 5. Outlier Detection: '''
# Box plot for outlier detection
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['age', 'avg_glucose_level', 'bmi']])
plt.title('Box Plot for Outlier Detection')
plt.show()

''' 6. Feature Distribution Analysis: '''
# Histogram for feature distribution
data['age'].hist(bins=30, color='blue', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

''' 7. Data Types and Unique Value Counts: '''
# Data types and unique value counts
print("\nData Types:\n", data.dtypes)
print("\nUnique Value Counts:\n", data.nunique())

''' 9. Grouped Aggregations: '''
# Grouped aggregations by gender
grouped_data = data.groupby('gender')['stroke'].mean()
print("\nGrouped Aggregation by Gender:\n", grouped_data)

''' 10. Insights from Relationships Between Features: '''
# Pairwise analysis
sns.pairplot(data, hue='stroke')
plt.title('Pairwise Analysis of Features')
plt.show()

'''
PART #2:
Data Preprocessing:
'''

''' 1. Handle Missing Values: '''
# Fill missing values with the mean for numerical columns
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())  # Assign back to the column

''' 2. Encode Categorical Variables: '''
# Convert categorical variables to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)  # This will handle all categorical variables

''' 3. Scale or Normalize Numerical Features: '''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_encoded[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(data_encoded[['age', 'avg_glucose_level', 'bmi']])

''' 4. Split the Dataset: '''
from sklearn.model_selection import train_test_split

X = data_encoded.drop('stroke', axis=1)  # Ensure 'stroke' is dropped from features
y = data_encoded['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
PART #3:
Machine Learning Model:
'''

''' 1. Logistic Regression: '''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Make Prediction
y_pred = model.predict(X_test)

# Evaluate Model
print('\nAccuracy: ', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))