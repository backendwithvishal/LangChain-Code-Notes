from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

chat = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

result = chat.invoke("What is the langchain")

print(result.content)