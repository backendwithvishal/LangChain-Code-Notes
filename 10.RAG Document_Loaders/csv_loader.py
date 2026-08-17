# Import CSVLoader from LangChain community package to load CSV data
from langchain_community.document_loaders import CSVLoader

# Initialize the loader with the path to the CSV file
loader = CSVLoader(file_path = 'Social_Network_Ads.csv')

# Load the CSV content into a list of Document objects (each row becomes one document)
docs = loader.load()

# Print the total number of documents (rows) loaded
print(len(docs))

# Print the contents of the 70th document (index 69)
print(docs[69])