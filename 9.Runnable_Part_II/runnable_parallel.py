# Demonstration of executing multiple LCEL chains concurrently using RunnableParallel
from langchain_groq import ChatGroq  # Import ChatGroq model class
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate class
from langchain_core.output_parsers import StrOutputParser  # Import StrOutputParser class
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env
from langchain_core.runnables.base import RunnableSequence, RunnableParallel  # Import RunnableSequence and RunnableParallel

load_dotenv()  # Load environment variables (e.g. GROQ_API_KEY)

# Prompt 1: Generate a short Tweet post about topic
prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

# Prompt 2: Generate a professional LinkedIn post about topic
prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic}",
    input_variables=['topic']
)

# Initialize ChatGroq AI model instance
model = ChatGroq(
    model="llama-3.1-8b-instant"
)

# StrOutputParser to convert AI response object to string
parser = StrOutputParser()

# RunnableParallel executes both chains simultaneously:
# - "tweet": Runs prompt1 -> model -> parser
# - "Linkedin": Runs prompt2 -> model -> parser
parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "Linkedin": RunnableSequence(prompt2, model, parser)
})

# Invoke parallel chain with topic 'AI'
result = parallel_chain.invoke({"topic": "AI"})

# Print dictionary output containing both generated Tweet and LinkedIn post
print(result)