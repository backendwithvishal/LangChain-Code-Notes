# Demonstration of conditional logic flow in LangChain using RunnableBranch and RunnableSequence
from langchain_groq import ChatGroq  # Import ChatGroq model class
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate class
from langchain_core.output_parsers import StrOutputParser  # Import StrOutputParser class
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda  # Import Runnable classes

load_dotenv()  # Load environment variables (e.g. GROQ_API_KEY)

# First Prompt: Request detailed report generation
prompt1 = PromptTemplate(
    template='Write a detaile report on  {topic}',
    input_variables=['topic']
)

# Second Prompt: Request text summarization
prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

# Initialize ChatGroq AI model instance
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

# StrOutputParser to convert AI response object to string
parser = StrOutputParser()

# Initial chain: Generates detailed report string from prompt1
report_gen_chain = RunnableSequence(prompt1, model, parser)

# Conditional Branch chain using RunnableBranch:
# - Condition: IF word count of report > 200 words, run prompt2 summarization chain
# - Default/Fallback: ELSE pass report through unchanged (RunnablePassthrough)
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

# Combine report generation chain and conditional branch chain into final sequence
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Invoke final chain with input topic
print(final_chain.invoke({'topic': 'Russia vs Ukraine war'}))