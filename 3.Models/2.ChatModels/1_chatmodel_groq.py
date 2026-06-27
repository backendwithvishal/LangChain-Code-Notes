from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

chat = ChatGroq(model="llama-3.3-70b-versatile")

result = chat.invoke("What is the capital of Japan?")

print(result.content)