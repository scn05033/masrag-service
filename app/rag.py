# app/rag.py
import os, json, hashlib, re
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from .utils import split_into_chunks

CSV_PATH = os.getenv("CSV_PATH", "./data/lyrics.csv")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/distiluse-base-multilingual-cased-v2")

TOP_K = int(os.getenv("TOP_K", "3"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.40"))  # cosine similarity 임계(0~1)

class RAGStore:
    def __init__(self):
        self.embedder = None
        self.chroma_client = None
        self.collection = None
        self.csv_fingerprint_path = "./.csv_fingerprint.json"

    def _file_fingerprint(self, path: str):
        st = os.stat(path)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        return {"size": st.st_size, "mtime": int(st.st_mtime), "sha16": h}

    def _load_fp(self):
        if os.path.exists(self.csv_fingerprint_path):
            return json.load(open(self.csv_fingerprint_path))
        return None

    def _save_fp(self, fp):
        with open(self.csv_fingerprint_path, "w") as f:
            json.dump(fp, f)

    def load_or_build(self, force_rebuild: bool = False):
        if self.embedder is None:
            self.embedder = SentenceTransformer(EMBED_MODEL)

        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

        if not os.path.exists(CSV_PATH):
            raise RuntimeError(f"CSV not found: {CSV_PATH}")
        new_fp = self._file_fingerprint(CSV_PATH)
        old_fp = self._load_fp()

        # 컬렉션 존재 여부
        has_collection = False
        try:
            self.chroma_client.get_collection("lyrics_rag")
            has_collection = True
        except:
            pass

        should_rebuild = force_rebuild or (not has_collection) or (old_fp != new_fp)

        if should_rebuild:
            self._build_index(new_fp)
        else:
            self.collection = self.chroma_client.get_collection("lyrics_rag")
            print("[RAG] loaded existing collection")

    def _build_index(self, new_fp):
        print("[RAG] (re)building index...")
        df = pd.read_csv(CSV_PATH)

        # 가사 컬럼 추정
        candidates = [c for c in ["lyrics","lyric","가사","본문","text","content","lyric_text"] if c in df.columns]
        if candidates:
            col = candidates[0]
        else:
            text_cols = [c for c in df.columns if df[c].dtype == object]
            lens = {c: df[c].astype(str).str.len().mean() for c in text_cols}
            col = max(lens, key=lens.get)

        texts = df[col].astype(str).str.replace("\r","\n").str.strip().tolist()

        # 청크 생성
        ids, docs, metas = [], [], []
        for i, t in enumerate(texts):
            for j, ch in enumerate(split_into_chunks(t, 300, 80)):
                ids.append(f"doc_{i}_chunk_{j}")
                docs.append(ch)
                metas.append({"doc_id": f"doc_{i}"})

        # 새 컬렉션
        try:
            self.chroma_client.delete_collection("lyrics_rag")
        except: pass
        self.collection = self.chroma_client.create_collection(
            name="lyrics_rag",
            metadata={"hnsw:space": "cosine"}  # distance=1-cosine_similarity
        )

        # 업서트
        B = 1024
        for s in range(0, len(docs), B):
            batch_docs = docs[s:s+B]
            batch_ids  = ids[s:s+B]
            batch_meta = metas[s:s+B]
            vecs = self.embedder.encode(batch_docs, convert_to_numpy=True)
            self.collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=vecs)

        self._save_fp(new_fp)
        print(f"[RAG] built: {len(docs)} chunks")

    def build_query(self, emotion_label: str, mood_words: List[str], scene_summary: str, genre_hint: str, tempo_hint: str) -> str:
        mw = ", ".join(mood_words or [])
        return f"emotion:{emotion_label} genre:{genre_hint} tempo:{tempo_hint} mood:{mw} scene:{scene_summary}"

    def retrieve_contexts(self, query_text: str, top_k: int = TOP_K, min_sim: float = SIM_THRESHOLD) -> List[str]:
        qvec = self.embedder.encode([query_text], convert_to_numpy=True)
        res = self.collection.query(query_embeddings=qvec, n_results=top_k)
        docs = res["documents"][0]
        dists = res["distances"][0]  # cosine distance
        sims = [1.0 - d for d in dists]
        filtered = [d for d, s in zip(docs, sims) if s >= min_sim][:top_k]
        return filtered
