# Demonstration of passing intermediate outputs untouched using RunnablePassthrough
from langchain_groq import ChatGroq  # Import ChatGroq model class
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate class
from langchain_core.output_parsers import StrOutputParser  # Import StrOutputParser class
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env
from langchain_core.runnables import (RunnableSequence, RunnableParallel, RunnablePassthrough)  # Import Runnable classes

load_dotenv()  # Load environment variables (e.g. GROQ_API_KEY)

# First Prompt: Generate joke on topic
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Initialize ChatGroq AI model instance
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

# StrOutputParser to convert AI response object to string
parser = StrOutputParser()

# Second Prompt: Explain generated joke
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# Step 1 chain: Generates joke string from prompt1
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# Step 2 parallel chain:
# - 'joke': Keeps generated joke string untouched using RunnablePassthrough()
# - 'explanation': Feeds joke string into prompt2 -> model -> parser chain
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

# Combine Step 1 and Step 2 into final sequence
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Invoke final chain with input topic 'cricket'
print(final_chain.invoke({'topic': 'cricket'}))
