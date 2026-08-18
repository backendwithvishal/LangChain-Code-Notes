# Demonstration of using ChatGroq LLM model in LangChain to answer questions
from langchain_groq import ChatGroq # Import the ChatGroq class from the langchain_groq module
from dotenv import load_dotenv # Import the load_dotenv function from the dotenv module
load_dotenv() # Load environment variables from a .env file

llm = ChatGroq(model="llama-3.3-70b-versatile") # Create an instance of the ChatGroq class with the specified model

result = llm.invoke("What is the capital of Maharashtra?") # Invoke the model with the specified prompt

print(result.content) # Print the content of the response