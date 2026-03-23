from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.rag_engine import rag_engine
import os

app = FastAPI(
    title="MaintainIQ Industrial RAG API",
    description="Advanced predictive maintenance system using Hybrid RAG (Dense + Sparse Fusion).",
    version="0.1.0"
)

# Initialize data on boot
try:
    rag_engine.ingest_data()
except Exception as e:
    print(f"Data ingestion warning: {e}")

class QueryRequest(BaseModel):
    query: str

os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("app/static/index.html", "r") as f:
        return f.read()

@app.post("/ask")
def query_manuals(req: QueryRequest):
    result = rag_engine.query(req.query)
    return result

if __name__ == "__main__":
    import uvicorn
    # Use port 8002 to avoid conflicts with AutoQuant (8000) and NeuroStream (8001)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)
