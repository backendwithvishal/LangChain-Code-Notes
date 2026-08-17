# Import DirectoryLoader and PyPDFLoader to load multiple PDF files from a folder
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Configure the loader to read all PDF files from the 'books' directory
loader = DirectoryLoader(
    path = 'books',         # Directory containing the files
    glob = '*.pdf',         # Matching pattern to pick only PDF files
    loader_cls = PyPDFLoader # Loader class used to process each PDF file
)

# lazy_load() reads documents one by one on-demand to save memory
docs = loader.lazy_load()

# Print the metadata (like file source and page number) for each loaded document page
for document in docs:
    print(document.metadata)