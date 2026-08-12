from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import ( RunnableSequence, RunnableParallel, RunnablePassthrough )

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)


model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(
    prompt1,
    model,
    parser
)

linkedin_chain = RunnableSequence(
    prompt2,
    model,
    parser
)

parallel_chain = RunnableParallel({
    "joke": joke_gen_chain,
    "explanation": linkedin_chain
})

final_chain = RunnableSequence(
    RunnablePassthrough(),
    parallel_chain
)

result = final_chain.invoke({"topic": "Cricket"})

print(result["joke"])

print(result["explanation"])