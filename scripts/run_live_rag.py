from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.config import load_config
from backend.rag.database import VectorDatabase
from backend.llm.gemini_client import GeminiLLMClient
from backend.rag.normal_rag import NormalRAG
from backend.compressor.pipeline import PipelineComponents
from backend.rag.smart_rag import SmartRAG
from backend.embeddings.local_models import SentenceTransformersCrossEncoder, SentenceTransformersEmbedder

def main():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set. Please set it before running this script.")
        sys.exit(1)
        
    cfg = load_config()
    # Build standard components
    device = cfg.system.device if hasattr(cfg, "system") else "cpu"
    model_name = cfg.retriever.embedding_model if hasattr(cfg, "retriever") else "sentence-transformers/all-MiniLM-L6-v2"
    ce_model = cfg.compressor.cross_encoder_model if hasattr(cfg, "compressor") else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    print("Loading embedding and cross-encoder models...")
    embedder = SentenceTransformersEmbedder(model_name=model_name, device=device)
    cross_encoder = SentenceTransformersCrossEncoder(model_name=ce_model, device=device)
    
    db = VectorDatabase(embedder=embedder)
    fixtures = sorted((ROOT / "datasets" / "demo_company" / "documents").rglob("*.md"))
    fixtures = [p for p in fixtures if "SYNTHETIC DEVELOPMENT" in p.read_text(encoding="utf-8")]
    for f in fixtures:
        db.add_document(f.stem, f.read_text(encoding="utf-8"))
        
    print(f"Loaded database with {len(db)} chunks.")
    
    components = PipelineComponents(
        embedder=embedder,
        cross_encoder=cross_encoder,
    )
    
    print("Initializing Gemini Client...")
    llm = GeminiLLMClient(cfg.llm)
    
    # Report resolved model
    resolved_model = llm._resolved_model_name
    print(f"\nResolved Gemini model to use: '{resolved_model}'")
    
    question = "When was NovaCloud founded and where is its headquarters?"
    
    print("\nExecuting Normal RAG query...")
    try:
        normal = NormalRAG(db, llm, cfg).run(question)
        normal_ok = normal.succeeded
        normal_ans = normal.answer
        normal_err = normal.error
    except Exception as e:
        normal_ok = False
        normal_ans = ""
        normal_err = str(e)
        
    print(f"Normal RAG Success: {normal_ok}")
    if normal_ok:
        print(f"Normal RAG Answer: {normal_ans[:150]}...")
    else:
        print(f"Normal RAG Error: {normal_err}")
        
    print("\nExecuting Smart RAG query...")
    try:
        smart = SmartRAG(db, llm, cfg, components=components).run(question)
        smart_ok = smart.succeeded
        smart_ans = smart.answer
        smart_err = smart.error
    except Exception as e:
        smart_ok = False
        smart_ans = ""
        smart_err = str(e)
        
    print(f"Smart RAG Success: {smart_ok}")
    if smart_ok:
        print(f"Smart RAG Answer: {smart_ans[:150]}...")
    else:
        print(f"Smart RAG Error: {smart_err}")
        
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"- Selected Model: {resolved_model}")
    print(f"- Normal RAG Status: {'SUCCESS' if normal_ok else 'FAILED'}")
    print(f"- Smart RAG Status: {'SUCCESS' if smart_ok else 'FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
