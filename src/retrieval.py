from sentence_transformers import CrossEncoder
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import os
chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
chroma_client = chromadb.PersistentClient(path=chroma_path)                                            
collection = chroma_client.get_or_create_collection(                                                        #creating embedding storage
    name="pubmed_diabetes_collection", 
    embedding_function=SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5"))   #using BAAI/bge-small-en-v1.5 embedding model

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_rerank(query:str, n_results: int=20) -> list[dict]:
    """Retrieve top 20 related document from ChromaDB based on bi-encoder and rerank using cross-encoder to return top 5 results"""
    #Retrieve first 20 related results using bi-encoder
    initial_results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    retrieved_docs = initial_results['documents'][0]    #Extracting documents from retrieved results
    metadatas = initial_results['metadatas'][0]
    pairs = [[query, doc] for doc in retrieved_docs] 
    scores = cross_encoder.predict(pairs)               #Cross-encoding score with query and doc pair
    reranked_results = sorted(zip(retrieved_docs, scores, metadatas), key=lambda x: x[1], reverse=True)

    return [
        {"text": doc, "score": float(score), "pmid": meta["pmid"], "title": meta["title"], "date": meta["date"]}
        for doc, score, meta in reranked_results[:5]
    ]

if __name__ == "__main__":
    results = retrieve_rerank("metformin side effects")
    for r in results:
        print(f"\n[{r['score']:.4f}] {r['title']}")
        print(f"  PMID: {r['pmid']}")