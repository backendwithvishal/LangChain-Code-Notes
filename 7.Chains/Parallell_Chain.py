# Import required libraries from LangChain and dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load secret environment variables (like API keys) from the .env file
load_dotenv()

# Step 1: Initialize Hugging Face AI Model
# Model 1: Qwen 2.5 7B hosted on Hugging Face
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational"
)
hf_model = ChatHuggingFace(llm=llm)

# Step 2: Initialize Groq AI Model
# Model 2: Llama 3.3 70B hosted on Groq
groq_model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Step 3: Define Prompt Templates for parallel tasks
# Task A prompt: Generate short study notes from text
prompt1 = PromptTemplate(
    template="Generate short and single notes from the following text:\n{text}",
    input_variables=["text"]
)

# Task B prompt: Generate 5 quiz question-answer pairs from text
prompt2 = PromptTemplate(
    template="Generate 5 short question-answer pairs from the following text:\n{text}",
    input_variables=["text"]
)

# Task C prompt: Merge the generated notes and quiz into one document
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

# Define output parser to get plain text output
parser = StrOutputParser()

# Step 4: Define Parallel Execution (RunnableParallel)
# RunnableParallel runs Task A ("notes") and Task B ("quiz") at the SAME time!
# Both tasks receive the same input text simultaneously.
parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | hf_model | parser,
        "quiz": prompt2 | groq_model | parser,
    }
)

# Step 5: Define the Merging Chain
# Takes the combined outputs {"notes": ..., "quiz": ...} from parallel execution
# and feeds them into prompt3 + groq_model to generate the combined document.
merge_chain = prompt3 | groq_model | parser

# Step 6: Connect Parallel Chain and Merge Chain
# Flow: Input Text -> Parallel Execution (Notes + Quiz simultaneously) -> Merge Chain -> Final Output
chain = parallel_chain | merge_chain

# Sample input text about Support Vector Machines (SVMs)
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

# Step 7: Run (Invoke) the entire parallel & merge chain
result = chain.invoke({"text": text})

# Step 8: Print the final combined document
print(result)

# Step 9: Display the visual workflow diagram in ASCII text
chain.get_graph().print_ascii()