# Demonstration of building sequential execution chains explicitly using RunnableSequence
from langchain_groq import ChatGroq  # Import ChatGroq model class
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate class
from langchain_core.output_parsers import StrOutputParser  # Import StrOutputParser class
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env
from langchain_core.runnables import RunnableSequence  # Import RunnableSequence class

load_dotenv()  # Load environment variables (e.g. GROQ_API_KEY)

# First Prompt: Generate information text on topic
prompt1 = PromptTemplate(
    template="Write information about {topic}",
    input_variables=["topic"]
)

# Initialize ChatGroq AI model instance
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

# StrOutputParser to convert AI response object to string
parser = StrOutputParser()

# Second Prompt: Explain text output from prompt1
prompt2 = PromptTemplate(
    template='Explain the following text - {text} ',
    input_variables=['text']
)

# Construct sequential pipeline explicitly using RunnableSequence:
# Step 1: prompt1 -> model -> parser (produces text string)
# Step 2: prompt2 -> model -> parser (takes text string as input)
chain = RunnableSequence(
    prompt1,
    model,
    parser,
    prompt2,
    model,
    parser
)

# Invoke full sequential pipeline starting with topic 'AI'
print(chain.invoke({"topic": "AI"}))
