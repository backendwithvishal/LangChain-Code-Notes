# Demonstration of manually invoking multiple prompts sequentially without LCEL output parsers
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # Import Hugging Face integrations
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate class

load_dotenv()  # Load environment variables (e.g. Hugging Face API key)

# Initialize Hugging Face Endpoint specifying model repository and task
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

# Wrap Hugging Face endpoint in ChatHuggingFace model
model = ChatHuggingFace(llm=llm)

# First Prompt: Instructs LLM to write a detailed report on a given topic
template = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Second Prompt: Instructs LLM to summarize text into 5 lines
template1 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

# Format the first prompt template with the topic "AI Engineering"
prompt = template.invoke({"topic": "AI Engineering"})

# Generate detailed report from LLM model
result = model.invoke(prompt)

# Pass the generated report content into the second prompt template
prompt1 = template1.invoke({"text": result.content})

# Generate 5-line summary from LLM model
result1 = model.invoke(prompt1)

# Print the final summarized content to console
print(result1.content)