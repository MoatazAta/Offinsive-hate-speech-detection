import streamlit as st
from Backend import *


def app():
    st.title('Offensive/hate Speech Detection')
    st.write("In our website you can check the content of a tweet and our job is to classify it into "
             "hate speech, offensive language or neither.")
    tweet = st.text_input("Tweet Text", "")
    if st.button("Submit"):
        result = prediction(tweet)
        st.write(result)

app()