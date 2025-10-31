# app/utils.py
import re

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def ngram_set(s: str, n: int = 3) -> set:
    s = normalize_text(s)
    toks = s.split()
    return set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))

def overlap_ratio(a: str, b: str, n: int = 3) -> float:
    A, B = ngram_set(a, n), ngram_set(b, n)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def split_into_chunks(text: str, max_chars: int = 300, min_chars: int = 80):
    """빈 줄 기준 단락 → 너무 길면 max_chars로 자르기."""
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text or "") if p.strip()]
    chunks = []
    for p in parts:
        if len(p) <= max_chars:
            if len(p) >= min_chars:
                chunks.append(p)
        else:
            s = 0
            while s < len(p):
                e = min(s + max_chars, len(p))
                piece = p[s:e].strip()
                if len(piece) >= min_chars:
                    chunks.append(piece)
                s = e
    return chunks

def short(text: str, max_chars: int = 2000) -> str:
    return (text or "")[:max_chars]

def clean_lyrics_output(text: str) -> str:
    t = text or ""
    # 코드블록/라벨/메타 설명 제거
    t = t.replace("```", "")
    # 흔한 메타 문장들 제거
    t = re.sub(r"^.*?다시 작성했습니다\.\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*\[(?:Emotion|Logic|Situation|Finalizer|Rewrite).*?\]\s*$", "", t, flags=re.MULTILINE|re.IGNORECASE)
    t = re.sub(r"^\s*(아래 '수정 지시'|수정 지시|지침|가이드라인).*?$", "", t, flags=re.MULTILINE)
    # verse:, chorus:, 설명 라벨 제거
    t = re.sub(r"^\s*(verse|chorus)\s*:\s*", "", t, flags=re.MULTILINE|re.IGNORECASE)
    # 공백 정리
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t
