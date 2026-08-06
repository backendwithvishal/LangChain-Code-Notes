from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational"
)

# Chat Model
hf_model = ChatHuggingFace(llm=llm)

# Groq Model
groq_model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate short and single notes from the following text:\n{text}",
    input_variables=["text"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="Generate 5 short question-answer pairs from the following text:\n{text}",
    input_variables=["text"]
)

# Final Prompt
prompt3 = PromptTemplate(
    template="""
Merge the provided notes and quiz into a single document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | hf_model | parser,
        "quiz": prompt2 | groq_model | parser,
    }
)

merge_chain = prompt3 | groq_model | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates.

The support vector machines in scikit-learn support both dense and sparse sample vectors as input.
"""

result = chain.invoke({"text": text})

print(result)