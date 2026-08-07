# Import required tools from LangChain and dotenv
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load secret environment variables (like API keys) from the .env file
load_dotenv()

# Step 1: Define the Prompt Template
# This creates a reusable template where {topic} will be replaced by user input
prompt = PromptTemplate(
    template = "Generate 5 interesting facts about {topic}",
    input_variables = ["topic"]
)

# Step 2: Initialize the AI Model
# Connecting to Groq using the Llama 3.3 70B model
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Step 3: Define the Output Parser
# Converts the raw AI response object into a clean string text
parser = StrOutputParser()

# Step 4: Build the Simple Chain using the pipe operator (|)
# Workflow: Prompt -> AI Model -> Output Parser
chain = prompt | model | parser 

# Step 5: Run (Invoke) the Chain
# Pass the topic value to run the entire pipeline
result = chain.invoke({'topic' : 'Indian Indenpendace Day'})

# Step 6: Print the final answer received from the AI
print(result)

# Step 7: Display the workflow graph in ASCII format
graph = chain.get_graph().print_ascii()