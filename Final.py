import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Apply a dull green theme
st.markdown(
    """
    <style>
    .main { background-color: #d8e6d3; } /* Dull green background */
    </style>
    """,
    unsafe_allow_html=True
)

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("stroke-data.csv")
    return data

# Preprocess dataset
@st.cache_data
def preprocess_data(data):
    data['bmi'] = data['bmi'].fillna(data['bmi'].mean())  # Handle missing values
    data_encoded = pd.get_dummies(data, drop_first=True)  # Encode categorical variables
    scaler = StandardScaler()
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])  # Scale numerical data
    return data_encoded

# Train model
@st.cache_data
def train_model(data_encoded):
    X = data_encoded.drop('stroke', axis=1)
    y = data_encoded['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=5000, solver='lbfgs', class_weight='balanced')  # Handle class imbalance and increase iterations
    model.fit(X_train, y_train)
    return model, X.columns, X_test, y_test, X_train, y_train

# Load and preprocess data
data = load_data()
data_encoded = preprocess_data(data)
model, feature_columns, X_test, y_test, X_train, y_train = train_model(data_encoded)

# Streamlit App
st.title("Stroke Prediction Dashboard")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Choose a Section:", ["Introduction", "EDA", "Model", "Conclusion"])

if menu == "Introduction":
    st.header("Introduction")
    st.write(
        "This project aims to analyze stroke data to identify key factors contributing to strokes "
        "and develop a predictive model to estimate stroke risks. "
        "We utilize exploratory data analysis (EDA) techniques to gain insights into the dataset "
        "and train a logistic regression model to make predictions. "
        "The dataset contains various features such as age, BMI, glucose levels, and categorical factors like gender and health conditions. "
        "Below is a preview of the dataset:"
    )
    st.dataframe(data.head())

elif menu == "EDA":
    st.header("Exploratory Data Analysis")

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Count plot of strokes
    st.subheader("Stroke Count Plot")
    fig, ax = plt.subplots()
    sns.countplot(x="stroke", data=data, ax=ax, palette="Greens")
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    correlation = data_encoded.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="Greens", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Boxplot for outliers
    st.subheader("Outlier Detection")
    fig, ax = plt.subplots()
    sns.boxplot(data=data[['age', 'avg_glucose_level', 'bmi']], ax=ax, palette="Greens")
    st.pyplot(fig)

    # Age distribution histogram
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    data['age'].hist(bins=30, ax=ax, color='green', alpha=0.7)
    st.pyplot(fig)

    # Pairwise analysis
    st.subheader("Pairwise Analysis")
    fig = sns.pairplot(data, hue="stroke", diag_kind="kde", palette="Greens")
    st.pyplot(fig)

    # Gender proportion by stroke
    st.subheader("Gender Proportion by Stroke")
    gender_stroke = data.groupby(['gender', 'stroke']).size().unstack().fillna(0)
    fig, ax = plt.subplots()
    gender_stroke.div(gender_stroke.sum(axis=1), axis=0).plot(kind="bar", stacked=True, colormap="Greens", ax=ax)
    st.pyplot(fig)

    # Age vs Glucose scatter plot
    st.subheader("Age vs. Glucose Levels")
    fig, ax = plt.subplots()
    sns.scatterplot(x="age", y="avg_glucose_level", hue="stroke", data=data, palette="Greens", ax=ax)
    st.pyplot(fig)

elif menu == "Model":
    st.header("Model and Prediction")

    # Display model results
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.text(classification_report(y_test, y_pred))

    # Prediction form
    st.subheader("Predict Stroke Risk")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
    avg_glucose_level = st.slider("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)

    # Process user inputs
    user_data = pd.DataFrame({
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "gender_Male": [1 if gender == "Male" else 0],
        "gender_Other": [1 if gender == "Other" else 0]
    })

    # Align input columns with trained model features
    for col in feature_columns:
        if col not in user_data.columns:
            user_data[col] = 0
    user_data = user_data[feature_columns]

    # Predict
    if st.button("Predict"):
        prediction = model.predict(user_data)[0]
        if prediction == 1:
            st.error("The model predicts **high risk of stroke**. Consult a medical professional.")
        else:
            st.success("The model predicts **low risk of stroke**.")

    # Example inputs to test
    st.subheader("Test Inputs for Validation")
    st.write("Example Input 1: Gender=Male, Age=45, Hypertension=Yes, Heart Disease=No, Glucose=120, BMI=28")
    st.write("Example Input 2: Gender=Female, Age=60, Hypertension=No, Heart Disease=Yes, Glucose=150, BMI=35")

elif menu == "Conclusion":
    st.header("Conclusion")
    st.write(
        "This project analyzed stroke data using EDA techniques and trained a logistic regression model. "
        "The key insights include the importance of age, glucose levels, and BMI in predicting stroke risks. "
        "The logistic regression model, with balanced class weights, achieved reasonable accuracy. "
        "Further improvements could involve testing more complex models like Random Forest or Gradient Boosting."
    )
