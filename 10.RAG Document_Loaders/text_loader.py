from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model = 'llama-3.3-70b-versatile'
)

prompt = PromptTemplate(
    template = 'Write a summary for the following poem - \n {poem}',
    input_variables = ['poem']
)

loader = TextLoader('cricket.txt', encoding='utf-8')
docus = loader.load()

# print(len(docus))

# print(type(docus))

# print(docus[0])

print(docus[0].page_content)

print(docus[0].metadata)