# Interactive command-line chatbot built with LangChain and ChatGroq
import os  # Import os module for reading environment variables
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env
from langchain_groq import ChatGroq  # Import ChatGroq integration class

# Load environment variables (e.g. GROQ_API_KEY) from .env file
load_dotenv()

# Initialize the Groq AI chat model instance
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Start continuous interactive conversation loop
while True:
    user_input = input("You: ")  # Get prompt text from user console input
    if user_input == "exit":  # Check if user typed 'exit' to stop the chatbot
        break
    result = model.invoke(user_input)  # Send user message to the AI model

    print("AI: ", result.content)  # Display AI response content on console