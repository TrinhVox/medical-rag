import json
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


ids = []
documents = []
metadatas = []

data_folder = Path("../data")
for file in data_folder.iterdir():                          #extracting data
    with open(str(file), "r") as f:
        record = json.load(f)
        if not record.get("AB"): 
            continue
        ids.append(record["PMID"])                          #unique identifier
        documents.append(record["TI"] + " " + record["AB"]) #embedding chunk
        metadatas.append(
            {
                "pmid": record["PMID"],
                "title": record["TI"],
                "authors": " ".join(record["AU"]),
                "date": record["DP"],
            }
        )

#embedding data
chroma_client = chromadb.PersistentClient(path="../chroma_db")                                              
collection = chroma_client.get_or_create_collection(                                                        #creating embedding storage
    name="pubmed_diabetes_collection", 
    embedding_function=SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5"))   #using BAAI/bge-small-en-v1.5 embedding model

collection.add(ids=ids, documents=documents, metadatas=metadatas)

results = collection.query(query_texts=["metformin side effects"], n_results=5)
for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- Result {i+1} ---")
    print(doc[:200])


