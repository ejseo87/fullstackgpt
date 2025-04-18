import streamlit as st
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")
st.title(today)
model = st.selectbox("select a number", ("GPT-4", "GPT-3.5"))

if model == "GPT-3.5":
    st.write("cheap")
else:
    st.write("expensive")
    name = st.text_input("Enter your name")
    st.write(name)


    value = st.slider("temperature", min_value=0.1, max_value=1.0)
    st.write(value)

    if st.button("Submit"):
        st.write(f"Hello {name}")



