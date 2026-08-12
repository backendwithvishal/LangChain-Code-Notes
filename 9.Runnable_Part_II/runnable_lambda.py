from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import ( RunnableSequence, RunnableParallel, RunnablePassthrough )

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

parser = StrOutputParser()