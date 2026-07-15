import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

st.header("Research Tool")

user_input = st.text_input("Enter your Topic of Interest:")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)