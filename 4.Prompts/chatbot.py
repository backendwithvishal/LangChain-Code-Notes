from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key="GROQ_API_KEY"
)

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    result = model.invoke(user_input)

    print("AI: ", result.content)

response = model.invoke()