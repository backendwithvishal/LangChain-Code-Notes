from langchain_core.prompts.loading import _load_prompt
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Create the LLM
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

st.header("📄 Research Tool 🔍")

# Research paper selection
paper_input = st.selectbox(
    "Select Research Paper", ["Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Synthesis",]) 

# Explanation style
style_input = st.selectbox(
    "Explanation Style",["Beginner-Friendly","Technical","Code-Oriented","Mathematical",] )

# Explanation length
length_input = st.selectbox(
    "Explanation Length",["Short (1-2 paragraphs)","Medium (3-5 paragraphs)","Long (Detailed Explanation)",])

# prompt
template = _load_prompt('template.json')

# calling the LLM model
if st.button("Summarize"):
    Chain = template | model
    # fill the placeholder
    result = Chain.invoke({
        "paper_input":paper_input,
        "style_input":style_input,
        "length_input":length_input,
    })
    st.write(result.content)