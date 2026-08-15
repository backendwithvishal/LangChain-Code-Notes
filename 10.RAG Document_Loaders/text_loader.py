from langchain_community.document_loaders import TextLoader

loader = TextLoader('cricket.txt', encoding='utf-8')
docus = loader.load()
# print(docus)
print(type(docus))
# print(docus[0].page_content)