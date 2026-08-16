from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('')

docs = loader.load()

print(docs)