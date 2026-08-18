# Demonstration of setting up a Groq chat model with custom temperature setting
from langchain_groq import ChatGroq # Import the ChatGroq class from the langchain_groq module
from dotenv import load_dotenv # Import the load_dotenv function from the dotenv module
load_dotenv() # Load environment variables from a .env file

chat = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5) # Create an instance of the ChatGroq class with the specified model and temperature

result = chat.invoke("What is the langchain") # Invoke the model with the specified prompt

print(result.content) # print the content from the result