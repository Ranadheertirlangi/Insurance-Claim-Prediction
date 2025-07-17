import pandas as pd
from sklearn.linear_model import LogisticRegression

import streamlit as st
from pickle import load
from pickle import dump

st.title("Model Deployment: pre-built Logistic Regression")

st.sidebar.header("User input parameters")

def user_input():
    sex = st.sidebar.selectbox('Gender', ('0','1'))
    insur = st.sidebar.selectbox('Insurance', ('0','1'))
    seatbelt = st.sidebar.selectbox('Seat Belt', ('0','1'))
    age = st.sidebar.number_input('Age')
    loss = st.sidebar.number_input("Insert Loss")
    
    data = {'CLMSEX': sex,
            'CLMINSUR': insur,
            'SEATBELT': seatbelt,
            'CLMAGE': age,
            'LOSS': loss}
    
    a = pd.DataFrame(data, index=[0])
    return a

st.subheader('User input')
df = user_input()
st.write(df)

load_model = load(open('Model.sav', 'rb'))  # load model from disk

pred = load_model.predict(df)
pred_proba = load_model.predict_proba(df)

st.subheader('Prediction Result')
st.write('Yes , The person will hire an attorney' if pred_proba[0][1] >0.54 else 'No')

st.subheader('Prediction probability')
st.write(pred_proba)