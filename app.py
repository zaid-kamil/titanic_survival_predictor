# in the terminal
# streamlit run app.py
import streamlit as st
import numpy as np
from joblib import load

def load_model():
    model = load('titanic_knn_classifier_v1.joblib')
    return model

st.title('Titanic Predictor')
st.image('header.jpg')
with st.form("input form"):
    ship_lvl = st.selectbox("select ship level",[1,2,3])
    age = st.number_input('select age of person',
                            max_value=100,
                            min_value=1,
                            value=25)
    gender = st.radio('select gender of person',['male','female'])
    is_alone= st.checkbox('is the person alone',value=False)
    sibsp = st.selectbox('how many children along the person',list(range(0,9)))
    parent = st.selectbox('are parent along the person',list(range(0,6)))
    st.form_submit_button('predict')
# load the model components
m = load_model()
scaler = m.get('scaling')
clf = m.get('classifier')
# preprocess the user input
gender = 0 if gender=='male' else 1
alone = 0 if is_alone==False else 1
# packing the data in array
userinput = np.array([ship_lvl,gender,age,sibsp,parent,alone])
# scale if neccessary
userinput = scaler.transform([userinput])
# predict and show respone
out = clf.predict(userinput)
if out[0] == 0:
    st.success('this person would not survive😢')
else:
    st.success('this person would survive⚓')