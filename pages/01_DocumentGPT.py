import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon=":memo:",
)
st.title("DocumentGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
        if save:
            st.session_state["messages"].append(
                {"role": role, "message": message})


for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)


message = st.chat_input("Send a message to the AI")
if message:
    send_message(message, "human", save=True)
    time.sleep(2)
    send_message("You said: " + message, "ai", save=True)
