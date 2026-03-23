🚀 **Final Portfolio Project Deployed: MaintainIQ!** ⚙️

Standard RAG (Retrieval-Augmented Generation) has a massive problem in Industrial IoT applications: semantic vector search routinely misses the *exact* Engineering Error Codes. 

To solve this for my concluding Capstone AI project, I built **MaintainIQ** — an advanced Predictive Maintenance AI that abandons traditional Naive RAG in favor of **Reciprocal Rank Fusion (RRF)**!

Instead of relying solely on heavy cloud vector databases, MaintainIQ runs an offline Hybrid Search engine bridging:
1️⃣ **Dense Semantic Search (FAISS):** Understands the contextual meaning of a technician's query.
2️⃣ **Sparse Keyword Search (BM25):** Employs the classic Okapi algorithm to guarantee it NEVER misses rigorous exact-match error codes (like "PRV-3" or "Error 404").

These two completely isolated retrieval systems run in parallel, and their output scoring is fused mathematically via our custom RRF algorithm before being passed into the LLM context window. The result? Zero-hallucination diagnostics pulling directly from complex engineering manuals.

I also built an industrial-themed Dark Dashboard UI right into the FastAPI layer so you can visually see the Fusion contexts being extracted in real-time.

Check out the architecture here: https://github.com/blackmangoo/MaintainIQ

#ArtificialIntelligence #MachineLearning #RAG #FAISS #FastAPI #DataScience #FASTNU #IndustrialIoT #SoftwareEngineering #TechPortfolio
