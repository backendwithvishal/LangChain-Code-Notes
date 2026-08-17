# Import required modules from LangChain and dotenv
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file (e.g., API keys)
load_dotenv()

# Initialize the ChatGroq model with Llama-3.3 70B model
model = ChatGroq(
    model = 'llama-3.3-70b-versatile'
)

# Define a prompt template that asks the LLM to summarize a poem
prompt = PromptTemplate(
    template = 'Write a summary for the following poem - \n {poem}',
    input_variables = ['poem']
)

# Output parser to parse the LLM output into a plain string
parser = StrOutputParser()

# Load the text file 'cricket.txt' using UTF-8 encoding
loader = TextLoader('cricket.txt', encoding='utf-8')
docus = loader.load()

# Print text content of the loaded document
print(docus[0].page_content)

# Print metadata of the loaded document (like file path)
print(docus[0].metadata)

# Create an LCEL (LangChain Expression Language) chain connecting prompt, model, and output parser
chain = prompt | model | parser

# Run the chain passing the poem text as input and print the summary response
print(chain.invoke({'poem': docus[0].page_content}))