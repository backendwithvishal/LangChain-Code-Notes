# Import required modules for LangChain, Pydantic, and dotenv
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import format_instructions
from langchain_openrouter import ChatOpenRouter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# Load secret environment variables (like API keys) from the .env file
load_dotenv()

'''
Optional Hugging Face Endpoint setup (commented out):

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational"
)

Chat Model
hf_model = ChatHuggingFace(llm=llm) 
'''

# Step 1: Initialize AI Models
# Groq Model using Llama 3.3 70B
groq_model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# OpenRouter Model using Llama 3.3 70B Instruct with temperature=0 for consistent responses
openrouter_model = ChatOpenRouter(
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0
)

# Standard text output parser
parser = StrOutputParser()

# Step 2: Define Data Structure for Sentiment Analysis using Pydantic
# This enforces that the AI model output MUST be one of: 'positive', 'negative', or 'neutral'
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative','neutral'] = Field(description = 'Give the sentiment of the feedback' )

# Create a parser specifically for the Feedback class
parser1 = PydanticOutputParser(pydantic_object = Feedback)

# Step 3: Define Classifier Prompt Template
# Asks the model to classify feedback sentiment and injects formatting instructions so output matches our Pydantic schema
prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instructions}",
    input_variables = ["feedback"],
    partial_variables = {'format_instructions':parser1.get_format_instructions()}
)

# Step 4: Build Classifier Chain
# Workflow: Input Feedback -> OpenRouter AI -> Pydantic Parser (returns structured Feedback object with .sentiment)
classifier_chain = prompt1 | openrouter_model | parser1

# Step 5: Define Response Prompts based on Sentiment
# Prompt for positive feedback
prompt2 = PromptTemplate(
    template = 'Write an appropiate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)

# Prompt for negative feedback
prompt3 = PromptTemplate(
    template = 'Write an appropiate response to this negative feedback \n {feedback}',
    input_variables = ['feedback']
)

# Step 6: Build Conditional Branch Chain (RunnableBranch)
# Works like IF / ELSE IF / ELSE logic:
# - IF sentiment == 'positive': Run positive response prompt -> OpenRouter -> Output Parser
# - ELSE IF sentiment == 'negative': Run negative response prompt -> Groq -> Output Parser
# - ELSE: Return fallback message "could not classify the feedback"
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | openrouter_model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | groq_model | parser),
    RunnableLambda(lambda x: "could not classify the feedback")
) 

# Step 7: Combine Classifier Chain and Branch Chain into one main pipeline
# Workflow: Sentiment Classification -> Conditional Routing -> Appropriate Response Generation
chain = classifier_chain | branch_chain

# Step 8: Test the chain with sample user feedback
print(chain.invoke({'feedback': 'this is a good phone'}))

# Step 9: Display the visual workflow graph in ASCII text
graph = chain.get_graph().print_ascii()