from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loader = TextLoader('./main.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

query = "What phrase should we say?"
index.query_with_sources(query)