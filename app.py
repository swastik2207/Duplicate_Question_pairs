import streamlit as st
import helper
import pickle

model = pickle.load(open('Question_model.pkl','rb'))

st.header('Duplicate Questions Detector')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button("Check"):
    query = helper.query_creator(q1,q2)
    result = model.predict(query)[0]

    if result:
        st.subheader('Duplicate')
    else:
        st.subheader('Not Duplicate')