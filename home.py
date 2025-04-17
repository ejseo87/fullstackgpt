import streamlit as st
from langchain.prompts import PromptTemplate
a = [1, 2, 3, 4]
d = {"a": 1, "b": 2}
p = PromptTemplate.from_template("xxxxx")
a
d
p
st.selectbox("select a number", ("GPT-4", "GPT-3.5"))
