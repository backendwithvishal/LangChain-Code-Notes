from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

res = llm.invoke("What is the capital of Maharashtra?")

print(res.content)