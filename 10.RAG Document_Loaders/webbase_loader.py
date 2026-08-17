import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

# Set user agent for the web request
os.environ["USER_AGENT"] = "Mozilla/5.0"


# Create the Groq model
model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# Create the prompt
prompt = PromptTemplate(
    template="""
Answer the question using the following webpage content.

Question:
{question}

Webpage Content:
{text}
""",
    input_variables=["question", "text"]
)


# Convert the model response into a string
parser = StrOutputParser()


# Simple public webpage
url = "https://www.langchain.com/"


# Load the webpage
loader = WebBaseLoader(url)

docs = loader.load()


# Create the chain
chain = prompt | model | parser


# Ask a question about the webpage
response = chain.invoke({
    "question": "What is LangChain?",
    "text": docs[0].page_content
})


print(response)