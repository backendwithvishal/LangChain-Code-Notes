from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

class Review(TypedDict):
    summery: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    "The hardware is great but the battery is not good"
)

print(type(result))