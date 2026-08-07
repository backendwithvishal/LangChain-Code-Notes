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

load_dotenv()
'''
Hugging Face Endpoint

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational"
)

Chat Model
hf_model = ChatHuggingFace(llm=llm) '''

# Groq Model

groq_model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# OpenRouter Model

openrouter_model = ChatOpenRouter(
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0
)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative','neutral'] = Field(description = 'Give the sentiment of the feedback' )

parser1 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instructions}",
    input_variables = ["feedback"],
    partial_variables = {'format_instructions':parser1.get_format_instructions()}
)

classifier_chain = prompt1 | openrouter_model | parser1

prompt2 = PromptTemplate(
    template = 'Write an appropiate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropiate response to this negative feedback \n {feedback}',
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | openrouter_model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | groq_model | parser),
    RunnableLambda(lambda x: "could not classify the feedback")
) 

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'this is a good phone'}))