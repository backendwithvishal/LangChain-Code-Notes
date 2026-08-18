# Demonstration of connecting to Hugging Face Inference API using ChatHuggingFace
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # Import Hugging Face classes from LangChain integration
from dotenv import load_dotenv  # Import load_dotenv to load API keys from .env file

load_dotenv()  # Load environment variables (e.g., HUGGINGFACEHUB_API_TOKEN)

# Initialize the Hugging Face Endpoint specifying the open-source model repository
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

# Wrap the endpoint with ChatHuggingFace to interact with it as a chat model
model = ChatHuggingFace(llm=llm)

# Send a prompt to the model and store the response
result = model.invoke("What is the capital of India")

# Print the generated text response content to console
print(result.content)