# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openrouter import ChatOpenRouter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()
'''
Hugging Face Endpoint

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational"
)

Chat Model
hf_model = ChatHuggingFace(llm=llm) '''

# Groq Model

groq_model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# OpenRouter Model

openrouter_model = ChatOpenRouter(
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0
)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = """
    Review the following feedback and determine if it is positive or negative Feedback: {Feedback}""",
    input_variables = ["Feedback"]
)

print(prompt1)