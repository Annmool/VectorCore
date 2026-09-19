"""
Server entrypoint for VectorCore Mini Vector Database and RAG application.
"""

import uvicorn
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("=" * 60)
    print(" VectorCore: Mini Vector Database & RAG Engine")
    print(" Built From Scratch with HNSW, IVF, BM25, RRF & Citations")
    print("=" * 60)
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    print(f" Server running at: http://{host}:{port}")
    print(f" API Documentation: http://{host}:{port}/docs")
    print("=" * 60)
    uvicorn.run("app.api:app", host=host, port=port, reload=False)
