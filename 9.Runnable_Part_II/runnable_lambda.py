# Demonstration of wrapping custom Python functions into LCEL pipelines using RunnableLambda
from langchain_groq import ChatGroq  # Import ChatGroq model class
from langchain_core.prompts import PromptTemplate  # Import PromptTemplate class
from langchain_core.output_parsers import StrOutputParser  # Import StrOutputParser class
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from .env
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel  # Import Runnable components

load_dotenv()  # Load environment variables (e.g. GROQ_API_KEY)

# Custom Python function to count words in a string
def word_count(text):
    return len(text.split())

# Prompt Template instructing model to generate a joke
prompt = PromptTemplate(
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

# Initial chain: Generates joke string from prompt
joke_gen_chain = RunnableSequence(prompt, model, parser)

# Parallel chain using RunnableParallel:
# - 'joke': Passes generated joke string through untouched using RunnablePassthrough
# - 'word_count': Executes custom Python word_count function wrapped inside RunnableLambda
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

# Combine joke generation chain and parallel processing chain into final sequence
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Invoke final chain with input topic 'AI'
result = final_chain.invoke({'topic': 'AI'})

# Format and combine output joke and word count
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

# Print final result
print(final_result)
