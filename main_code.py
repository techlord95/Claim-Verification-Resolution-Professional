# updated_gemini_genai_rag.py
import os
import asyncio
from typing import List, Dict, Any, Optional
import uuid
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from google import genai
from dotenv import load_dotenv
import chardet

load_dotenv()
# from google.genai.types import EmbedContentConfig  # used for embedding config (optional)

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from langchain.text_splitter import RecursiveCharacterTextSplitter

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiEmbeddings:
    """
    Embedding wrapper using the new google-genai SDK:
      - Uses genai.Client().models.embed_content for batch embeddings
      - Falls back to SentenceTransformer local model on errors
    """

    def __init__(self,
                 gemini_api_key: Optional[str] = None,
                 model_name: str = "gemini-embedding-001",
                 fallback_model: str = "all-MiniLM-L6-v2"):
        # initialize genai client (it will also pick up GEMINI_API_KEY / GOOGLE_API_KEY if unset)
        try:
            if gemini_api_key:
                self.client = genai.Client(api_key=gemini_api_key)
            else:
                self.client = genai.Client()  # picks up env var if present
            self.use_genai = True
            logger.info("Using google-genai client for embeddings.")
        except Exception as e:
            logger.warning("Could not initialize genai client; will use local fallback. %s", e)
            self.client = None
            self.use_genai = False

        self.model_name = model_name
        self.fallback_model_name = fallback_model
        # placeholder dimension — will set once we obtain an embedding
        self.dim = None

        # Lazy-load fallback model only if needed
        self._st_model = None

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts. Attempts genai client; on any failure uses SentenceTransformer as fallback."""
        if self.use_genai and self.client:
            try:
                # call embed_content in batch (client.models.embed_content supports list of contents)
                # set config if you want output dimensionality etc.
                # cfg = EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    # config=cfg
                )
                embeddings = self._extract_embeddings_from_response(response)
                # set dim if unknown
                if embeddings and self.dim is None:
                    self.dim = len(embeddings[0])
                return embeddings
            except Exception as e:
                logger.warning("genai embed_content failed; falling back to SentenceTransformer. %s", e)
                self.use_genai = False  # avoid trying again in future (unless you want retries)
                # fallthrough to fallback
        # fallback to SentenceTransformer
        return await self._fallback_embed(texts)

    async def embed_query(self, text: str) -> List[float]:
        emb = await self.embed_documents([text])
        return emb[0] if emb else [0.0] * (self.dim or 768)

    def _extract_embeddings_from_response(self, response) -> List[List[float]]:
        """
        The SDK may return pydantic models or dict-like objects.
        This function extracts embeddings robustly from common shapes:
         - response.embeddings -> list of objects with .values
         - response['embeddings'] -> list of dicts with 'values'
        """
        embeddings = []
        # Try attribute access first
        if hasattr(response, "embeddings") and response.embeddings:
            for item in response.embeddings:
                # item may have .values attribute
                if hasattr(item, "values"):
                    embeddings.append(list(item.values))
                elif isinstance(item, dict) and "values" in item:
                    embeddings.append(item["values"])
                else:
                    # fallback: try to find any list-like numeric field
                    vals = getattr(item, "values", None) or item.get("values") if isinstance(item, dict) else None
                    if vals:
                        embeddings.append(list(vals))
        else:
            # Try dict-like access
            try:
                raw = dict(response)
            except Exception:
                raw = None
            if raw and "embeddings" in raw:
                for item in raw["embeddings"]:
                    if isinstance(item, dict) and "values" in item:
                        embeddings.append(item["values"])
        # final sanity: ensure numeric lists
        cleaned = []
        for v in embeddings:
            cleaned.append([float(x) for x in v])
        return cleaned

    async def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        if self._st_model is None:
            logger.info("Loading SentenceTransformer fallback model: %s", self.fallback_model_name)
            self._st_model = SentenceTransformer(self.fallback_model_name)
            self.dim = self._st_model.get_sentence_embedding_dimension()
        # run encode in thread so we don't block event loop
        return await asyncio.to_thread(self._st_model.encode, texts, show_progress_bar=False)


class HybridRAGSystem:
    def __init__(self, gemini_api_key: Optional[str] = None, qdrant_url: str = "http://localhost:6333",
                 qdrant_api_key: Optional[str] = None, collection_name: str = "resolution_documents"):
        # embeddings provider (will use gemini-genai if available)
        self.emb = GeminiEmbeddings(gemini_api_key=gemini_api_key)
        # qdrant init
        if qdrant_api_key:
            self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.qdrant = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        # text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # faiss placeholders
        self.faiss_index = None
        self.document_chunks = []
        # create collection with dynamic dim when possible
        self._ensure_qdrant_collection()

    def _ensure_qdrant_collection(self):
        # Try to create collection if not exists. Use a safe default dim (768) if unknown.
        dim = self.emb.dim or 3072  # gemini-embedding-001 commonly returns 3072; fallback 768/3072
        try:
            cols = self.qdrant.get_collections().collections
            if not any(c.name == self.collection_name for c in cols):
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
                logger.info("Created Qdrant collection %s with dim=%d", self.collection_name, dim)
            else:
                logger.info("Qdrant collection %s exists", self.collection_name)
        except Exception as e:
            logger.warning("Could not ensure qdrant collection on init: %s", e)

    # async def load_documents(self, md_paths: List[str]):
    #     all_chunks = []
    #     for p in md_paths:
    #         pth = Path(p)
    #         if not pth.exists():
    #             logger.warning("Missing file: %s", p)
    #             continue
    #         if pth.suffix.lower() not in {".md", ".markdown", ".txt"}:
    #             logger.warning("Skipping non-markdown file: %s", p)
    #             continue
    #         text = pth.read_text(encoding="utf-8")
    #         chunks = self.text_splitter.split_text(text)
    #         for i, c in enumerate(chunks):
    #             all_chunks.append({"content": c, "source": str(pth), "chunk_id": f"{pth.stem}_{i}", "metadata": {"path": str(pth)}})

    #     if not all_chunks:
    #         logger.info("No chunks to process.")
    #         return
        
    async def load_documents(self, md_paths: List[str]):
        all_chunks = []
        for p in md_paths:
            pth = Path(p)
            if not pth.exists():
                
                logger.warning("Missing file: %s", p)
                continue
            if pth.suffix.lower() not in {".md", ".markdown", ".txt"}:
                logger.warning("Skipping non-markdown file: %s", p)
                continue

    
            raw = pth.read_bytes()
            enc = chardet.detect(raw)["encoding"] or "utf-8"
            try:
                text = raw.decode(enc, errors="replace")
            except Exception as e:
                logger.error("Failed decoding %s with %s: %s", p, enc, e)
                continue
            logger.info("Loaded %s using %s encoding", p, enc)

            chunks = self.text_splitter.split_text(text)
            for i, c in enumerate(chunks):
                all_chunks.append({
                    "content": c,
                    "source": str(pth),
                    "chunk_id": f"{pth.stem}_{i}",
                    "metadata": {"path": str(pth)}
                })

        texts = [c["content"] for c in all_chunks]
        # get embeddings in batches using GeminiEmbeddings (which may call genai or fallback)
        batch_size = 64
        points = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            emb_batch = await self.emb.embed_documents(batch_texts)
            for i, emb in enumerate(emb_batch):
                idx = start + i
                ch = all_chunks[idx]
                pts = PointStruct(id=str(uuid.uuid4()), vector=list(emb),
                                  payload={"content": ch["content"], "source": ch["source"], "chunk_id": ch["chunk_id"], "metadata": ch["metadata"]})
                points.append(pts)
            # upsert in smaller batches into qdrant
            try:
                self.qdrant.upsert(collection_name=self.collection_name, points=points)
                logger.info("Upserted %d points to Qdrant", len(points))
                points = []
            except Exception as e:
                logger.error("Qdrant upsert error: %s", e)

        # final flush if any points left
        if points:
            try:
                self.qdrant.upsert(collection_name=self.collection_name, points=points)
            except Exception as e:
                logger.error("Final qdrant upsert failed: %s", e)

        # create FAISS backup index
        await self._create_faiss_index(all_chunks)

    async def _create_faiss_index(self, chunks: List[Dict[str, Any]]):
        texts = [c["content"] for c in chunks]
        embeddings = await self.emb.embed_documents(texts)
        arr = np.array(embeddings, dtype="float32")
        if arr.size == 0 or arr.ndim != 2:
            logger.warning("Invalid embeddings for FAISS; skipping.")
            return
        faiss.normalize_L2(arr)
        d = arr.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(arr)
        self.document_chunks = chunks
        logger.info("Created FAISS index with %d vectors (dim=%d)", arr.shape[0], d)

    async def generate_with_gemini(self, prompt: str, model: str = "gemini-1.5-pro"):
        """Use genai client for final completion (safe wrapper)."""
        if not self.emb.use_genai or not getattr(self.emb, "client", None):
            logger.warning("GenAI client not available; cannot call generate_content.")
            return None
        try:
            resp = self.emb.client.models.generate_content(model=model, contents=prompt)
            # response.text works in examples
            txt = getattr(resp, "text", None) or (resp[0].text if len(resp) > 0 else None)
            return txt
        except Exception as e:
            logger.error("generate_content failed: %s", e)
            return None


# Example main
async def main():
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    rag = HybridRAGSystem(gemini_api_key=GEMINI_KEY, qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    md_files = [r"scratch\2022-04-28-181717-r28jw-af0143991dbbd963f47def187e86517f.md",
                r"scratch\INSOLVENCY AND BANKRUPTCY BOARD OF INDIA (LIQUIDATION).md"]
    await rag.load_documents(md_files)
    # sample call to gemini generate (optional)
    out = await rag.generate_with_gemini("Summarize the key IBC timelines for an RP.", model="gemini-2.5-flash")
    print("Generated:", out)

if __name__ == "__main__":
    asyncio.run(main())
