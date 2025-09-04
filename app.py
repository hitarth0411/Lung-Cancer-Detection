import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Lung Cancer Detection")

# Load data
df = pd.read_csv("data.csv")

# If 'index' is a column (not the actual index), set it as the target
if 'index' in df.columns:
    y = df['index']
    df = df.drop(columns=['index'], errors='ignore')
elif 'Patient Id' in df.columns:
    y = df['Patient Id']
    df = df.drop(columns=['Patient Id'], errors='ignore')
elif 'ID' in df.columns:
    y = df['ID']
    df = df.drop(columns=['ID'], errors='ignore')
elif 'Patient_ID' in df.columns:
    y = df['Patient_ID']
    df = df.drop(columns=['Patient_ID'], errors='ignore')
else:
    st.error("Neither 'index' nor 'Patient Id' found as a column for target.")
    st.stop()

    # Drop index and patient_id if exists
df = df.drop(columns=['index','Patient_Id'], errors='ignore')

# Drop missing values
df = df.dropna()

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature matrix
X = df

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Sidebar input
st.sidebar.header("Patient Health Parameters")
user_input = {}
for col in X.columns:
    val = st.sidebar.number_input(
        f"{col}", 
        min_value=float(X[col].min()), 
        max_value=float(X[col].max()), 
        value=float(X[col].mean())
    )
    user_input[col] = val

# Predict
input_df = pd.DataFrame([user_input])
if st.sidebar.button("Predict"):
    prediction = knn.predict(input_df)[0]
    st.success(f"Prediction (Index/ID): {prediction}")
