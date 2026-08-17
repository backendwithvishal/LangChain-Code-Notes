# Import os module for managing environment variables
import os

# Import WebBaseLoader to scrape web pages and ChatGroq / prompt tools from LangChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load API keys and environment variables from .env file
load_dotenv()

# Set User-Agent header so web requests identify as a standard web browser
os.environ["USER_AGENT"] = "Mozilla/5.0"

# Initialize the ChatGroq model with the specified LLM
model = ChatGroq(
    model="openai/gpt-oss-20b"
)

# Create a prompt template instructing the model to answer questions based on webpage content
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

# Parser to transform the LLM output object into a plain string
parser = StrOutputParser()

# Specify the URL of the public webpage to load
url = "https://www.langchain.com/"

# Initialize WebBaseLoader to fetch HTML content from the specified URL
loader = WebBaseLoader(url)

# Scrape and parse the webpage content into Document objects
docs = loader.load()

# Create an LCEL processing chain (Prompt -> LLM -> Output Parser)
chain = prompt | model | parser

# Invoke the chain with a specific question and the scraped webpage text
response = chain.invoke({
    "question": "What is LangChain?",
    "text": docs[0].page_content
})

# Print the model's response to the console
print(response)