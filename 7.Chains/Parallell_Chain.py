from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

# LLM
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate short and single notes from the follwing text \n {text}",
    input_variables=["text"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="Generate a 5 short question answer from the following text:\n{text}",
    input_variables=["text"]
)

# Prompt 3
# Final Prompt
prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single doucument. \n notes -> {notes} and quiz -> {quiz}",
    input_variables = ["notes", "quiz"]
) 

parser = StrOutputParser()

