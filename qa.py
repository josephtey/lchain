import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)

while True:
  question = input("Question: ")
  result = chain({"question": question})
  print(f"Answer: {result['answer']}")
  print(f"All sources: {result['sources']}")

  main_source = result['sources'].split(",")[-1]
  main_source_file_name = main_source.split("<split>")[0]
  main_source_index = int(main_source.split("<split>")[1])
  corresponding_docstore = store.index_to_docstore_id[main_source_index]

  print(f"Source: {store.docstore.search(store.index_to_docstore_id[main_source_index]).page_content}")