# Import typing utilities for structured output schema definitions
from typing import Literal, TypedDict, Annotated, Optional
# Import ChatGroq integration from LangChain
from langchain_groq import ChatGroq
# Import load_dotenv to read environment variables from .env file
from dotenv import load_dotenv

# Load environment variables (such as GROQ_API_KEY) from .env file
load_dotenv()

# Initialize the Groq model instance with Llama 3.3 70B model
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Define a TypedDict output schema with field annotations to guide LLM response format
class Review(TypedDict):
    key_themes: Annotated[list[str], "List down all the key themes of the review which are generally related to product"]
    summery: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["Postive", "Negative", "Neutral"], "The sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write down the name of the reviewer"]

# Bind the TypedDict schema to the LLM to get structured dictionary output
structured_model = model.with_structured_output(Review)

# Send input review text to the LLM and receive structured data output matching the Review schema
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

# Print specific field 'name' from the extracted structured result
# print(result)
# print(["summary"])
print(result["name"])