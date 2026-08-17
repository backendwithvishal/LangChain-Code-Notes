# Import PyPDFLoader from LangChain to extract text and metadata from PDF files
from langchain_community.document_loaders import PyPDFLoader

# Initialize PyPDFLoader with the PDF file path
loader = PyPDFLoader('dl-curriculum.pdf')

# Load the entire PDF (each page becomes a separate document object)
docs = loader.load()

# Print total number of pages in the PDF document
print(len(docs))

# Print text content of the first page (index 0)
print(docs[0].page_content)

# Print metadata (such as page number and file source) of the second page (index 1)
print(docs[1].metadata)