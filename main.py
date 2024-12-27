import pandas as pd

#Load the DataSet:
data = pd.read_csv('F:\Data Science\Intro to Data Science\stroke-dataset-analysis\stroke-data.csv')
print(data.head())
print()

'''SummaRy Statistics:'''
print('Summary Statistics:\n', data.describe())

'''Visualizations:'''
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