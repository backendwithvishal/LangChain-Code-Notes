import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

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
    "Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

# Explanation style
style_input = st.selectbox(
    "Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical",
    ],
)

# Explanation length
length_input = st.selectbox(
    "Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (Detailed Explanation)",
    ],
)

if st.button("Summarize"):

    prompt = f"""
    Summarize the research paper "{paper_input}".

    Use a {style_input} explanation.

    Keep the explanation {length_input}.

    Explain the main idea, important concepts, advantages,
    limitations, and real-world applications.
    """

    response = model.invoke(prompt)

    st.subheader("Summary")
    st.write(response.content)