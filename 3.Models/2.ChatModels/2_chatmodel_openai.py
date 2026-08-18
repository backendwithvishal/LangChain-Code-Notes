# Demonstration of using ChatOpenAI class connected to Groq's OpenAI-compatible API endpoint
from langchain_groq import ChatGroq # Import the ChatOpenAI class
import os # Import os for fetching API keys from environment variables
from dotenv import load_dotenv # Import the load_dotenv function from the dotenv module
load_dotenv() # Load environment variables from a .env file

# Create an instance of the ChatOpenAI class targeting Groq's OpenAI-compatible API
chat = ChatGroq(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5
)

result = chat.invoke("What is the Node.js") # Invoke the model with the specified prompt

print(result.content) # print the content from the result