import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('F:/Data Science/Intro to Data Science/stroke-dataset-analysis/stroke-data.csv')
    return data

# Preprocess dataset
@st.cache_data
def preprocess_data(data):
    # Handle missing values
    data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

    # Fix categorical values
    if 'gender' in data.columns:
        data['gender'] = data['gender'].replace({'MaleFemale': 'Unknown'})

    # Encode categorical variables
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Scale numerical features
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    if all(col in data_encoded.columns for col in numerical_cols):
        data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])

    return data, data_encoded

# Train logistic regression model
@st.cache_data
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model

# Load and preprocess data
data = load_data()
raw_data, data_encoded = preprocess_data(data)

# Split dataset
X = data_encoded.drop('stroke', axis=1, errors='ignore')
y = data_encoded['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_model(X_train, y_train)
y_pred = model.predict(X_test)

# Streamlit Dashboard
st.title("Stroke Dataset Analysis Dashboard")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Section:", ["Introduction", "EDA", "Data Preprocessing", "Model Results"])

if menu == "Introduction":
    st.header("Introduction")
    st.write("Analyzing the Stroke Dataset using EDA, preprocessing, and predictive modeling.")
    st.dataframe(data.head())

elif menu == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values[missing_values > 0])

    st.subheader("Count Plot of Stroke")
    fig, ax = plt.subplots()
    sns.countplot(x='stroke', data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Matrix")
    correlation = data_encoded.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Box Plot for Outlier Detection")
    fig, ax = plt.subplots()
    sns.boxplot(data=data[['age', 'avg_glucose_level', 'bmi']], ax=ax)
    st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    data['age'].hist(bins=30, ax=ax, color='blue', alpha=0.7)
    st.pyplot(fig)

    st.subheader("Pairwise Analysis")
    st.write("This analysis shows pairwise relationships between features.")
    fig = sns.pairplot(data, hue="stroke", diag_kind="kde")
    st.pyplot(fig)

elif menu == "Data Preprocessing":
    
    st.header("Data Preprocessing")
    st.subheader("Encoded Data")
    st.dataframe(data_encoded.head())

elif menu == "Model Results":
    st.header("Model Results")

    st.subheader("Accuracy")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Feature Importances")
    feature_importances = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
    st.bar_chart(feature_importances)