import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

chain = VectorDBQAWithSourcesChain.from_chain_type(llm=OpenAI(temperature=0), vectorstore=store)

while True:
  question = input("Question: ")
  result = chain({"question": question})
  print(f"Answer: {result['answer']}")
  print(f"All sources: {result['sources']}")

  all_sources = result['sources'].replace(";", ",").split(",")
  all_docs = []

  for source in all_sources:
    file_name = source.split("<split>")[0]
    index = int(source.split("<split>")[1])
    corresponding_docstore = store.index_to_docstore_id[index]
    doc = store.docstore.search(corresponding_docstore).page_content

    all_docs.append(doc)

  for doc in all_docs:
     print(doc)
     print("\n")