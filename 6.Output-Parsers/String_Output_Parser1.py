# Demonstration of chaining multiple prompts and LLMs cleanly using LCEL and StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # Import Hugging Face model classes
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env file
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate class
from langchain_core.output_parsers import StrOutputParser  # Import StrOutputParser to convert response objects to strings

load_dotenv()  # Load environment variables (e.g. API keys)

# Initialize Hugging Face Endpoint specifying model repository and task
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

# Wrap Hugging Face endpoint in ChatHuggingFace model
model = ChatHuggingFace(llm=llm)

# First Prompt: Generate detailed report on topic
template = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Second Prompt: Generate 5-line summary from detailed report text
template1 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

# Initialize StrOutputParser to convert AI response object to string automatically
parser = StrOutputParser()

# Build LCEL chain: template -> model -> parser -> template1 -> model -> parser
# The string output from prompt 1 automatically feeds into prompt 2 ('text' input variable)
chain = template | model | parser | template1 | model | parser

# Invoke entire sequential chain starting with initial topic
result = chain.invoke({"topic": "AI Engineering"})

# Print the final summarized string result
print(result)