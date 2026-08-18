# Demonstration of generating document embeddings locally using HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  # Import HuggingFaceEmbeddings wrapper class

# Load a pre-trained open-source sentence transformer model locally for text embeddings
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define a list of text document strings to generate vector representations for
documents = [
    "Mumbai is the capital of Maharashtra",
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
]

# Convert the text documents into numerical vector embeddings
vector = embedding.embed_documents(documents)

# Print the resulting numerical embedding vectors as string output
print(str(vector))