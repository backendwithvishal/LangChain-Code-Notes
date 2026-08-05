from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [  
    ResponseSchema(name='news_headline',description='Headline of the news'),
    ResponseSchema(name='summary',description='2-3 line summary of the news'),
    ResponseSchema(name='sentiment',description='Sentiment of the news'),
    ResponseSchema(name='source',description='Source of the news')
]