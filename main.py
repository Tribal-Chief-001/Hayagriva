import uvicorn
from src.config import settings

if __name__ == "__main__":
    print(f"\n[Hayagriva] Starting RAG Server on http://localhost:{settings.PORT}")
    uvicorn.run("api.index:app", host="0.0.0.0", port=settings.PORT, reload=True)