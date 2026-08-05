from String_Output_Parser1 import parser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = 'Give me the name, age and city of the fictional person \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instruction()}
)
prompt = template.format()

print(prompt)