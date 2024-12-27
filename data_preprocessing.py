import pandas as pd

#Load the DataSet:
data = pd.read_csv('F:\Data Science\Intro to Data Science\stroke-dataset-analysis\stroke-data.csv')
print(data.head())

if __name__ == "__main__":
    print("Data loaded successfully.")