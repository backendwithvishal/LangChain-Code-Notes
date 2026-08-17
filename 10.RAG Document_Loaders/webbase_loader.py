from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Create the Groq model
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Create the prompt
# Use the same variable names everywhere
prompt = PromptTemplate(
    template="Answer the following question:\n{question}\n\nFrom the following text:\n{text}",
    input_variables=["question", "text"]
)

# Convert model output into a simple string
parser = StrOutputParser()

# Load the webpage
url = "https://www.flipkart.com/apple-macbook-air-m4-16-gb-256-gb-ssd-macos-sequoia-mc7a4hn-a/p/itmdd70ae2c75bc6?pid=COMH9ZWQ5G9QKEFQ&marketplace=FLIPKART&lid=LSTCOMH9ZWQ5G9QKEFQV8HOI1&pageUID=1785852362603"

loader = WebBaseLoader(url)

# Load the webpage content
docs = loader.load()

# Create the chain
chain = prompt | model | parser

# Get the webpage text and send it with the question
print(
    chain.invoke({
        "question": "what is the price of this product?",
        "text": docs[0].page_content
    })
)