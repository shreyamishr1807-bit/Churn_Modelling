import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
model=tf.keras.models.load_model("churn_model.h5")

# load scaler
with open("scaler.pkl","rb") as f:
    scaler=pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Enter customer details")

# user input
credit_score=st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=600
)
gender = st.selectbox(
    "Gender",
    ["Female","Male"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)
tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

balance = st.number_input(
    "Balance",
    value=60000
)
products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

credit_card = st.selectbox(
    "Has Credit Card",
    [0,1]
)

active_member = st.selectbox(
    "Is Active Member",
    [0,1]
)
salary = st.number_input(
    "Estimated Salary",
    value=50000
)


geography = st.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)


# Convert input into model format
if gender=="Male":
    gender_value=1
else:
    gender_value=0


# One hot encoding Geography
geo_germany = 1 if geography=="Germany" else 0
geo_spain = 1 if geography=="Spain" else 0


input_data = np.array([[
    credit_score,
    gender_value,
    age,
    tenure,
    balance,
    products,
    credit_card,
    active_member,
    salary,
    geo_germany,
    geo_spain
]])


# Scaling

input_scaled = scaler.transform(input_data)


# Prediction

if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.error("Customer is likely to leave the bank")
    else:
        st.success("Customer is likely to stay")