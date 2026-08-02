# Import ChatGroq integration from LangChain
from langchain_groq import ChatGroq
# Import load_dotenv to read environment variables from .env file
from dotenv import load_dotenv
# Import typing modules for field annotations
from typing import Literal, TypedDict, Annotated, Optional
# Import Pydantic tools to define structured data schemas
from pydantic import BaseModel, Field

# Load environment variables (such as GROQ_API_KEY) from .env file
load_dotenv()

# Initialize the Groq model instance with Llama 3.3 70B model
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Define a Pydantic BaseModel schema to specify structured output fields and descriptions
class Review(BaseModel):
    key_themes: list[str] = Field(description="List down all the key themes of the review which are generally related to product") 
    summery: str = Field(description="A brief summary of the review")
    sentiment: Literal["Postive", "Negative", "Neutral"] = Field(description="The sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write down the name of the reviewer")

# Bind the Pydantic schema to the model to guarantee structured response format
structured_model = model.with_structured_output(Review)

# Send input review text to the LLM and get a validated Pydantic object as output
result = structured_model.invoke( """ I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Aryan Sanam """
)

# Print the resulting structured Pydantic object
# print(result)
# print(["summary"])
print(result)