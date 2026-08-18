# Streamlit web application interface for generating research paper summaries using LangChain
import json  # Import json module for loading template configuration
import os  # Import os module for reading environment variables
import streamlit as st  # Import Streamlit library for web app UI
from dotenv import load_dotenv  # Import load_dotenv to load API keys
from langchain_core.prompts.loading import _load_prompt  # Helper function to load prompt object from dict
from langchain_groq import ChatGroq  # Import ChatGroq integration class

# Load environment variables (such as GROQ_API_KEY)
load_dotenv()

# Initialize ChatGroq AI model instance with Llama 3.3 70B versatile model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Display application header title in Streamlit UI
st.header("📄 Research Tool 🔍")

# Dropdown select box for choosing a research paper title
paper_input = st.selectbox(
    "Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

# Dropdown select box for choosing explanation style
style_input = st.selectbox(
    "Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical",
    ],
)

# Dropdown select box for choosing explanation length
length_input = st.selectbox(
    "Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (Detailed Explanation)",
    ],
)

# Load prompt template JSON configuration file
with open("template.json", "r", encoding="utf-8") as file:
    config = json.load(file)

# Deserialize JSON configuration into a LangChain PromptTemplate object
template = _load_prompt(config)

# Handle click event for the "Summarize" button
if st.button("Summarize"):
    # Build processing chain combining prompt template and AI model
    chain = template | model
    # Invoke chain with user selected options from UI controls
    result = chain.invoke(
        {
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input,
        }
    )
    # Render generated summary content in Streamlit UI
    st.write(result.content)