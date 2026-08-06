from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="Generate a 5 point summary from the following text:\n{text}",
    input_variables=["text"]
)

# Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

# Chat Model
model = ChatHuggingFace(llm=llm)

# Output Parser
parser = StrOutputParser()

# Sequential Chain
chain = prompt1 | model | parser | prompt2 | model | parser

# Invoke Chain
result = chain.invoke({
    "topic": "India's 80th Independence Year"
})

print(result)

graph = chain.get_graph().print_ascii()