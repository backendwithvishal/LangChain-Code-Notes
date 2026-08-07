# Import required tools from LangChain and dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load secret environment variables (like API keys) from .env file
load_dotenv()

# Step 1: Prompt Template for the First Stage
# Takes a {topic} and asks the AI to write a detailed report
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Step 2: Prompt Template for the Second Stage
# Takes the generated report text {text} from Stage 1 and asks for a 5-point summary
prompt2 = PromptTemplate(
    template="Generate a 5 point summary from the following text:\n{text}",
    input_variables=["text"]
)

# Step 3: Connect to Hugging Face Endpoint
# Uses the Qwen 2.5 7B model hosted on Hugging Face
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

# Step 4: Wrap the endpoint in a Chat Model wrapper
model = ChatHuggingFace(llm=llm)

# Step 5: Define the Output Parser
# Converts AI output into clean text strings
parser = StrOutputParser()

# Step 6: Build the Sequential Chain
# Sequential means step-by-step:
# Prompt 1 -> AI Model -> Output Text -> Prompt 2 (takes output text as input) -> AI Model -> Summary Text
chain = prompt1 | model | parser | prompt2 | model | parser

# Step 7: Run (Invoke) the Sequential Chain
# We only need to provide the initial {topic}. The output of prompt1 automatically feeds into prompt2!
result = chain.invoke({
    "topic": "India's 80th Independence Year"
})

# Step 8: Print the final summarized result
print(result)

# Step 9: Print an ASCII flow diagram of the sequential chain
graph = chain.get_graph().print_ascii()