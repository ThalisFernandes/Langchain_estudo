from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import ollama
from langchain.schema import Document
from typing import Optional



class SemanticCacheWrapper:
    def __init__(self, llm, threshold=0.8):
        self.llm=llm
        self.threshold=threshold
        self.embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorStore = Chroma(
            collection_name="Cache",
            embedding_function=self.embeddings,
            persist_directory="cache"
        )
    
    def invoke(self, prompt: str) -> Optional[str]:
        docs = self.vectorStore.similarity_search(prompt, k=1)

        if docs and docs[0].page_content == prompt:
            print("[Cache_hit]")
            return docs[0].metadata.get("response")
        
        print("[cache MISS]")

        response= self.llm.invoke(prompt)
        self.vectorStore.add_documents([Document(page_content=prompt, metadata={"response": response})])
        self.vectorStore.persist()

        return response