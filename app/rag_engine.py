import os
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# Load API keys
load_dotenv()

class HybridRAG:
    """ Advanced Information Retrieval Architecture using Reciprocal Rank Fusion """
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.bm25 = None
        
        # We ensure Google credentials fallback exists just like AutoQuant
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("OPENAI_API_KEY", "") 
        elif os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
            
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        except Exception as e:
            print("Warning: API Keys not configured yet.", e)
            self.embeddings = None
            self.llm = None
        
    def ingest_data(self, data_folder="data"):
        if not self.embeddings:
            return
            
        docs = []
        if os.path.exists(data_folder):
            for file in os.listdir(data_folder):
                if file.endswith(".txt"):
                    with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Simple chunking by paragraph for manuals
                        chunks = text.split('\n\n')
                        for i, chunk in enumerate(chunks):
                            if len(chunk.strip()) > 10:
                                docs.append(Document(page_content=chunk.strip(), metadata={"source": file, "chunk": i}))
        
        if not docs:
            docs = [Document(page_content="ERROR 404: Pump pressure exceeded 500 PSI. Immediate valve replacement required.", metadata={"source": "mock"})]
            
        self.documents = docs
        
        # 1. Initialize Dense Retrieval (FAISS - Semantic Meaning)
        try:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        except Exception as e:
            print("Dense Vector DB Failed. Falling back to pure Sparse BM25 Fusion.", e)
            self.vector_store = None
        
        # 2. Initialize Sparse Retrieval (BM25 - Exact Keyword Match)
        tokenized_corpus = [doc.page_content.lower().split(" ") for doc in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        # State-of-the-Art Fusion Algorithm bridging BM25 and Vector Search
        fused_scores = {}
        
        for rank, doc in enumerate(dense_results):
            doc_str = doc.page_content
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (k + rank)
            
        for rank, doc in enumerate(sparse_results):
            doc_str = doc.page_content
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (k + rank)
            
        # Sort and retrieve top contexts
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in sorted_docs[:3]]

    def query(self, question: str) -> dict:
        if not self.documents:
            self.ingest_data()
            
        if not self.llm:
            return {"answer": "Error: Configure GEMINI_API_KEY in .env", "retrieved_context": []}
            
        # 1. Dense Search
        if self.vector_store:
            dense_docs = self.vector_store.similarity_search(question, k=5)
        else:
            dense_docs = [] # Graceful fallback to pure sparse fusion
        
        # 2. Sparse Search
        tokenized_query = question.lower().split(" ")
        sparse_scores = self.bm25.get_scores(tokenized_query)
        top_n_idx = np.argsort(sparse_scores)[::-1][:5]
        sparse_docs = [self.documents[i] for i in top_n_idx]
        
        # 3. Hybrid Fusion (RRF)
        best_contexts = self.reciprocal_rank_fusion(dense_docs, sparse_docs)
        context_str = "\n---\n".join(best_contexts)
        
        # 4. LLM Generation
        prompt = f"You are MaintainIQ, an Industrial IoT diagnostic AI. Use the following engineering manual snippets to answer the technician's query precisely.\n\nContext:\n{context_str}\n\nQuery: {question}"
        
        try:
            response = self.llm.invoke(prompt)
            answer_text = response.content
        except Exception as e:
            print("LLM API Error encountered. Falling back to Mock Inference.", e)
            answer_text = (
                "MAINTAIN-IQ SYSTEM ALARM: \n\n"
                "Based on the mathematically extracted hybrid context shown below, your query relates to a critical failure protocol. "
                "Since live generative-inference API access is currently restricted, please strictly follow the procedures outlined in the 'Reciprocal Rank Fusion Extracted Contexts' safely."
            )
        
        return {
            "answer": answer_text,
            "retrieved_context": best_contexts
        }

rag_engine = HybridRAG()
