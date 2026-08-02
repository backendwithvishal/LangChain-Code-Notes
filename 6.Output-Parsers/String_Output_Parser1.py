from anyio import TemporaryDirectory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StringOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# First Prompt --> detailed report
template = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Second Prompt --> summary
template1 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

parser = StringOutputParser()

chain = template | model | parser | template1 | model | parser

result = chain.invoke({"topic": "AI Engineering"})

print(result)