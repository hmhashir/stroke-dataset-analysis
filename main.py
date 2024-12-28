import pandas as pd

#Load the DataSet:
data = pd.read_csv('F:\Data Science\Intro to Data Science\stroke-dataset-analysis\stroke-data.csv')
print(data.head())
print()

'''1. SummaRy Statistics:'''
print('Summary Statistics:\n', data.describe())

'''2. Visualizations:'''
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(style = 'Whitegrid')

#count plot for stroke:
plt.figure(figsize=(8, 6))
sns.countplot(x='stroke', data=data)
plt.title('Count of Strokes')
plt.xlabel('Stroke (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

'''3. Correlation Analysis:'''
# Convert categorical variables to numeric (if needed)
data_encoded = pd.get_dummies(data, columns=['gender'], drop_first=True)

# Select only numeric columns for correlation
data_numeric = data_encoded.select_dtypes(include=['number'])

correlation = data_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

'''4. Missing Value Analysis:'''
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values[missing_values > 0])

'''5. Outlier Detection:'''
#box plot for outlier detection
plt.figure(figsize = (10, 6))
sns.boxplot(data=data[['age', 'avg_glucose_level', 'bmi']])
plt.title('Box Plot for Outlier Detection')
plt.show()

'''6. Feature Distribution Analysis:'''
# Histogram for feature distribution
data['age'].hist(bins=30, color='blue', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()