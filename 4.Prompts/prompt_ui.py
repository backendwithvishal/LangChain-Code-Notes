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

st.header("📄 Research Tool")

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


template = PromptTemplate(
    template=""" Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length. """
) 

input_variables=["paper_input","style_input","length_input"]

# fill the placeholder
prompt = template.invoke({
    "paper_input":paper_input,
    "style_input":style_input,
    "length_input":length_input,
})

# calling the LLM model
if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)