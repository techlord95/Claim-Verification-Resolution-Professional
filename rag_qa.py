import os
import time
import pickle
import numpy as np
from typing import List, Tuple, Optional, Any
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
import faiss

from google import genai
from google.genai import types
import argparse
from dotenv import load_dotenv

load_dotenv()

# ==============================
# Setup Gemini client
# ==============================
api_key = os.environ.get("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key)


# ==============================
# Gemini embedding function (unchanged)
# ==============================
def gemini_embed_texts(
    texts: List[str],
    client: Optional[Any] = None,
    model: str = "models/text-embedding-004",
    batch_size: int = 64,
    max_retries: int = 3,
    output_dimensionality: Optional[int] = None,
    task_type: Optional[str] = None,
) -> List[np.ndarray]:
    if client is None:
        client = genai_client

    if not texts:
        return []

    vectors: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        attempt = 0
        while attempt < max_retries:
            try:
                config_dict = {}
                if output_dimensionality is not None:
                    config_dict["output_dimensionality"] = int(output_dimensionality)
                if task_type is not None:
                    config_dict["task_type"] = task_type

                config = None
                if config_dict:
                    try:
                        config = types.EmbedContentConfig(**config_dict)
                    except Exception:
                        config = None

                if config is not None:
                    res = client.models.embed_content(model=model, contents=batch, config=config)
                else:
                    res = client.models.embed_content(model=model, contents=batch)

                # Defensive parsing
                if hasattr(res, "embeddings"):
                    emb_src = res.embeddings
                elif isinstance(res, dict) and "embeddings" in res:
                    emb_src = res["embeddings"]
                else:
                    emb_src = list(res)

                for item in emb_src:
                    vec_vals = None
                    if item is None:
                        raise ValueError("Null embedding returned")

                    if hasattr(item, "embedding") and hasattr(item.embedding, "values"):
                        vec_vals = list(item.embedding.values)
                    elif hasattr(item, "values"):
                        vec_vals = list(item.values)

                    if vec_vals is None and isinstance(item, dict):
                        if "embedding" in item:
                            emb = item["embedding"]
                            if isinstance(emb, dict) and "values" in emb:
                                vec_vals = emb["values"]
                            elif isinstance(emb, list):
                                vec_vals = emb
                        elif "values" in item:
                            vec_vals = item["values"]

                        if vec_vals is None and "response" in item:
                            resp = item["response"]
                            if isinstance(resp, dict) and "embedding" in resp:
                                emb = resp["embedding"]
                                if isinstance(emb, dict) and "values" in emb:
                                    vec_vals = emb["values"]

                    if vec_vals is None and isinstance(item, (list, tuple, np.ndarray)):
                        vec_vals = list(item)

                    if vec_vals is None:
                        raise ValueError(f"Could not parse embedding: {repr(item)[:200]}")

                    vectors.append(np.array(vec_vals, dtype=np.float32))

                break  # success
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    raise
                time.sleep(0.6 * (2 ** (attempt - 1)))

    if not vectors:
        raise RuntimeError("No embeddings returned by Gemini")

    dims = {v.shape[0] for v in vectors}
    if len(dims) != 1:
        raise RuntimeError(f"Inconsistent embedding dims: {dims}")

    return [v.astype(np.float32) for v in vectors]


# ==============================
# PDF → text chunks (per-page chunking - original function kept)
# ==============================
def pdf_to_chunks_per_page(pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Extract text from PDF per-page and produce chunks that DO NOT cross page boundaries.
    Each chunk is prefixed with a small provenance tag: [PAGE n]
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1
            page_text = page.get_text("text") or ""
            page_text = "\n".join([line.strip() for line in page_text.splitlines() if line.strip()])
            if not page_text.strip():
                continue
            words = page_text.split()
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                slice_words = words[i: i + chunk_size]
                if not slice_words:
                    continue
                chunk_text = " ".join(slice_words)
                chunk_with_page = f"[PAGE {page_no}] {chunk_text}"
                chunks.append(chunk_with_page)

    print(f"Extracted {len(chunks)} chunks (per-page) from PDF '{os.path.basename(pdf_path)}'")
    return chunks


# ==============================
# PDF -> exactly one chunk per page (NEW)
# ==============================
def pdf_to_one_chunk_per_page(pdf_path: str) -> List[str]:
    """
    Create exactly one chunk per non-empty page.
    Each chunk is prefixed with provenance: [PAGE n]
    """
    chunks: List[str] = []
    page_word_counts: List[int] = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1
            page_text = page.get_text("text") or ""
            # Normalize whitespace and remove empty lines
            page_text = "\n".join([line.strip() for line in page_text.splitlines() if line.strip()])
            if not page_text.strip():
                page_word_counts.append(0)
                continue
            words = page_text.split()
            page_word_counts.append(len(words))
            # Keep full page as one chunk (no splitting)
            chunks.append(f"[PAGE {page_no}] {page_text}")

    # Diagnostic: print pages and average length
    non_empty_counts = [c for c in page_word_counts if c > 0]
    total_pages = len(page_word_counts)
    non_empty_pages = len(non_empty_counts)
    avg_words = (sum(non_empty_counts) / non_empty_pages) if non_empty_pages else 0
    print(f"PDF '{os.path.basename(pdf_path)}': {total_pages} pages, {non_empty_pages} non-empty pages.")
    print(f"Avg words / non-empty page: {avg_words:.1f}. Longest page: {max(non_empty_counts) if non_empty_counts else 0} words.")
    print(f"Extracted {len(chunks)} page-level chunks from PDF '{os.path.basename(pdf_path)}'")
    return chunks


# ==============================
# Save / Load indexes
# ==============================
def save_indexes(bm25: BM25Okapi, faiss_index: faiss.IndexFlatIP, chunks: List[str], prefix: str = "indexes/index"):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    bm25_file = f"{prefix}_bm25.pkl"
    chunks_file = f"{prefix}_chunks.pkl"
    faiss_file = f"{prefix}_faiss.index"

    with open(bm25_file, "wb") as f:
        pickle.dump(bm25, f)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)

    faiss.write_index(faiss_index, faiss_file)
    print(f"Saved BM25 -> {bm25_file}, CHUNKS -> {chunks_file}, FAISS -> {faiss_file}")


def load_indexes(prefix: str = "indexes/index") -> Tuple[BM25Okapi, faiss.IndexFlatIP, List[str]]:
    bm25_file = f"{prefix}_bm25.pkl"
    chunks_file = f"{prefix}_chunks.pkl"
    faiss_file = f"{prefix}_faiss.index"

    if not (os.path.exists(bm25_file) and os.path.exists(chunks_file) and os.path.exists(faiss_file)):
        raise FileNotFoundError("Index files not found for given prefix.")

    with open(bm25_file, "rb") as f:
        bm25 = pickle.load(f)
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)

    faiss_index = faiss.read_index(faiss_file)
    print(f"Loaded BM25 from {bm25_file}, CHUNKS from {chunks_file}, FAISS from {faiss_file}")
    return bm25, faiss_index, chunks


# ==============================
# Build (or load) indexes helper (modified to accept one_chunk_per_page)
# ==============================
def build_or_load_indexes(
    pdf_path: str,
    index_prefix: str = "indexes/index",
    chunk_size: int = 1000,
    overlap: int = 200,
    force_rebuild: bool = False,
    one_chunk_per_page: bool = False
) -> Tuple[BM25Okapi, faiss.IndexFlatIP, List[str]]:
    """
    If index files exist and force_rebuild is False -> load indexes.
    Otherwise, build indexes from PDF, save them, and return them.
    """
    bm25_file = f"{index_prefix}_bm25.pkl"
    chunks_file = f"{index_prefix}_chunks.pkl"
    faiss_file = f"{index_prefix}_faiss.index"

    if not force_rebuild and os.path.exists(bm25_file) and os.path.exists(chunks_file) and os.path.exists(faiss_file):
        try:
            return load_indexes(prefix=index_prefix)
        except Exception as e:
            print(f"Failed to load existing indexes ({e}), rebuilding...")

    # Use one-chunk-per-page if requested; otherwise use original per-page splitting
    if one_chunk_per_page:
        chunks = pdf_to_one_chunk_per_page(pdf_path)
    else:
        chunks = pdf_to_chunks_per_page(pdf_path, chunk_size=chunk_size, overlap=overlap)

    print("Building BM25 index...")
    tokenized = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    print("Generating embeddings for FAISS (this may take time)...")
    embeddings = gemini_embed_texts(
        chunks,
        client=genai_client,
        model="models/text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )

    X = np.vstack(embeddings).astype(np.float32)
    faiss.normalize_L2(X)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    save_indexes(bm25, index, chunks, prefix=index_prefix)
    return bm25, index, chunks


# ==============================
# Hybrid query (unchanged)
# ==============================
def hybrid_query(
    query: str,
    bm25: BM25Okapi,
    faiss_index: faiss.IndexFlatIP,
    chunks: List[str],
    top_k: int = 5,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5
) -> List[Tuple[str, float]]:
    print(f"Running hybrid query: '{query}'")
    bm25_scores = bm25.get_scores(query.split())
    max_bm = max(bm25_scores) if len(bm25_scores) > 0 else 0.0
    if max_bm > 0:
        bm25_scores = bm25_scores / max_bm
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k * 2]

    q_vec = gemini_embed_texts(
        [query],
        client=genai_client,
        model="models/text-embedding-004",
        task_type="RETRIEVAL_QUERY"
    )[0]

    q_vec = q_vec.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(q_vec)
    D, I = faiss_index.search(q_vec, top_k * 2)
    faiss_scores = D[0]
    faiss_top = I[0]

    final_scores = {}
    for idx in bm25_top:
        final_scores[idx] = final_scores.get(idx, 0) + (bm25_weight * float(bm25_scores[idx]))
    for i, idx in enumerate(faiss_top):
        final_scores[idx] = final_scores.get(idx, 0) + (semantic_weight * float(faiss_scores[i]))

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    results = [(chunks[idx], score) for idx, score in ranked[:top_k]]
    print(f"Found {len(results)} results")
    return results


# ==============================
# Generate answer (unchanged)
# ==============================
def generate_answer(query: str, context_chunks: List[Tuple[str, float]], client=None) -> str:
    if client is None:
        client = genai_client

    context = "\n\n".join([chunk for chunk, _ in context_chunks[:8]])
    prompt = f"""Based on the following context, please provide a comprehensive answer to the question.

Context:
{context}

Question: {query}

INSTRUCTIONS:
- Please provide a detailed answer based on the context provided.
- If the context doesn't contain enough information to fully answer the question, please indicate what information is available and what might be missing.
- if you get something in a tabular format , return the tables as it is
- Do not change the language mentioned in the context since it is a legal document , so not a word shall be changed while quoting from the context
- FOLLOW THESE VERY CAREFULLY AND REMEMBER THAT THIS IS A LEGAL DOCUMENT , NO CHANGE IN LANGUAGE SHOULD BE DONE

"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        if hasattr(response, "text"):
            return response.text
        elif isinstance(response, dict) and "text" in response:
            return response["text"]
        elif isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and "text" in response[0]:
                return response[0]["text"]
            return str(response)
        else:
            return str(response)
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ==============================
# Process multiple queries (interactive or from file)
# ==============================
def process_queries_loop(bm25, faiss_index, chunks, args):
    """
    Keeps indexes loaded in memory. Accepts:
      - interactive mode (type queries)
      - queries from a file (one per line)
      - single query via CLI
    """
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        if not os.path.exists(args.queries_file):
            print(f"Queries file not found: {args.queries_file}")
            return
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        # interactive loop
        print("Entering interactive query mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                q = input("\nQuery> ").strip()
                if not q:
                    continue
                if q.lower() in ("exit", "quit"):
                    print("Exiting interactive mode.")
                    return
                results = hybrid_query(q, bm25, faiss_index, chunks, top_k=args.top_k)
                print("\nTop Results (snippet):")
                for i, (chunk, score) in enumerate(results, 1):
                    print(f"\nResult {i} (score: {score:.4f}):")
                    print("-" * 60)
                    print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                if args.generate:
                    print("\nGenerating answer (LLM)...")
                    ans = generate_answer(q, results)
                    print("\n" + "=" * 60)
                    print("ANSWER:")
                    print("=" * 60)
                    print(ans)
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting interactive mode.")
                return

        return

    # If queries list provided (single or file), process them serially
    for q in queries:
        print(f"\nQuery: {q}")
        results = hybrid_query(q, bm25, faiss_index, chunks, top_k=args.top_k)
        print("\nTop Results (snippet):")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\nResult {i} (score: {score:.4f}):")
            print("-" * 60)
            print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
        if args.generate:
            print("\nGenerating answer (LLM)...")
            ans = generate_answer(q, results)
            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(ans)


# ==============================
# CLI / Main (added --one-chunk-per-page flag)
# ==============================
def parse_args():
    p = argparse.ArgumentParser(description="Hybrid RAG pipeline (per-page chunking + persisted indexes + multi-query session)")
    p.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
    p.add_argument("--index-prefix", type=str, default="indexes/index", help="Prefix for saved index files")
    p.add_argument("--chunk-size", type=int, default=2300, help="Chunk size (words)")
    p.add_argument("--overlap", type=int, default=2000, help="Chunk overlap (words)")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild indexes even if saved ones exist")
    p.add_argument("--one-chunk-per-page", action="store_true", help="Force exactly one chunk per PDF page (overrides chunk-size splitting).")
    p.add_argument("--query", type=str, help="Single query to run (non-interactive)")
    p.add_argument("--queries-file", type=str, help="File containing queries (one per line)")
    p.add_argument("--top-k", type=int, default=2, help="Number of top results to return")
    p.add_argument("--generate", action="store_true", help="Call LLM to generate final answer for each query")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        print("=== Starting Hybrid RAG Pipeline (per-page chunking + persisted indexes) ===")
        # BUILD / LOAD ONCE
        bm25, faiss_index, chunks = build_or_load_indexes(
            pdf_path=args.pdf,
            index_prefix=args.index_prefix,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            force_rebuild=args.rebuild,
            one_chunk_per_page=args.one_chunk_per_page
        )

        # PROCESS QUERIES (interactive or file or single)
        process_queries_loop(bm25, faiss_index, chunks, args)

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
