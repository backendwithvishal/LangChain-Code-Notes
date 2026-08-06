from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [  
    ResponseSchema(name='Fact_1',description='fact 1 above the topic'),
    ResponseSchema(name='Fact_2',description='fact 2 above the topic'),
    ResponseSchema(name='Fact_3',description='fact 3 above the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give me 3 fact about {topic} \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'Indian'})

result = model.invoke(prompt)

final_result = parser.parse(result)

print(final_result)