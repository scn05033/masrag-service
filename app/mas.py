# app/mas.py
import os, json
from google import genai
from .rag import RAGStore
from .utils import short, overlap_ratio

class MASRAG:
    def __init__(self, rag: RAGStore):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=key)
        self.rag = rag

    def llm(self, prompt: str, model="gemini-2.5-flash") -> str:
        """Gemini 호출 래퍼: 항상 str을 돌려주도록 방어."""
        try:
            resp = self.client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}]
            )
        except Exception as e:
            return f"[LLM_ERROR] {type(e).__name__}: {e}"

        # resp 자체가 None이거나, text가 없을 수 있음
        if not resp or getattr(resp, "text", None) is None:
            # 최소한 빈 문자열을 돌려서 .strip() 방지
            return ""

        return resp.text.strip()
    # -------- Planner --------
    def plan_guidelines(self, analysis: dict, extra_rules: str = "") -> dict:
        prompt = f"""
[Planner]
아래 정보를 바탕으로 가사 생성/리뷰 가이드라인을 만들어라.
- 감정: {analysis['emotion_label']}
- 장르: {analysis['genre_hint']}
- 템포: {analysis['tempo_hint']}
- 키워드: {", ".join(analysis['mood_words'])}
- 장면: {analysis['scene_summary']}

규칙:
1) 구조: verse(4줄) → chorus(4줄) → verse(4줄) → chorus(4줄)
2) 금지: 고유명사/저작권 의심 문구/비속어, 힌트 문구 복사 금지
3) 톤: 감정·장르·템포에 맞춘 어휘/리듬
4) 반복: 동일 구절 3회 이상 반복 금지(후렴 2회 이내 변형 허용)
5) 길이: 각 줄 4~14어절
{extra_rules}

아래 형식의 JSON만 출력(설명/라벨 금지):
{{
  "weights": {{"emotion": 0.4, "logic": 0.3, "situation": 0.3}},
  "banned_words": [],
  "must_have": [],
  "style": [],
  "structure": ["verse:4 lines", "chorus:4 lines", "repeat 2x max"],
  "checklist": ["감정 일치", "장면 합치", "구조 준수", "반복 과다 금지"]
}}
"""
        text = self.llm(prompt)
        defaults = {
            "weights": {"emotion": 0.4, "logic": 0.3, "situation": 0.3},
            "banned_words": [], "must_have": [], "style": [],
            "structure": ["verse:4 lines", "chorus:4 lines", "repeat 2x max"],
            "checklist": ["감정 일치", "장면 합치", "구조 준수", "반복 과다 금지"],
        }
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("planner output not object")
        except Exception:
            return {"raw": text, **defaults}
        for k, v in defaults.items():
            if k not in obj or obj[k] is None:
                obj[k] = v
        return obj

    # -------- Writer (with RAG) --------
    def writer(self, analysis: dict, top_k=3):
        query = self.rag.build_query(
            analysis["emotion_label"], analysis["mood_words"],
            analysis["scene_summary"], analysis["genre_hint"], analysis["tempo_hint"]
        )
        contexts = self.rag.retrieve_contexts(query, top_k=top_k)
        ctx_block = "\n".join(f"- {c}" for c in contexts)
        prompt = f"""
[Writer]
아래 '표현 힌트'는 영감용. **복사 금지**. 출력은 **가사 본문만**.
규칙: verse(4줄) → chorus(4줄) → verse(4줄) → chorus(4줄)
감정:{analysis['emotion_label']} 장르:{analysis['genre_hint']} 템포:{analysis['tempo_hint']}
장면:{analysis['scene_summary']} 키워드:{", ".join(analysis['mood_words'])}

[표현 힌트]
{short(ctx_block, 1500)}
"""
        draft = self.llm(prompt)
        return draft, contexts

    # -------- Reviewers --------
    def emotion_review(self, draft: str, analysis: dict, plan_json) -> str:
        plan_txt = json.dumps(plan_json, ensure_ascii=False) if isinstance(plan_json, dict) else (plan_json or "")
        prompt = f"""
[Emotion Reviewer]
가사와 가이드라인을 기준으로 감정/어휘 톤 일치 여부만 점검.
결과는 **'수정 지시'**만 간결히. 예시 최대 1줄.

[가이드라인]
{short(plan_txt, 1200)}

[가사]
{short(draft, 2000)}
"""
        return self.llm(prompt)

    def logic_review(self, draft: str, plan_json) -> str:
        plan_txt = json.dumps(plan_json, ensure_ascii=False) if isinstance(plan_json, dict) else (plan_json or "")
        prompt = f"""
[Logic Reviewer]
구조/줄 수/반복/금칙어 점검. 결과는 **'수정 지시'**만.

[가이드라인]
{short(plan_txt, 1200)}

[가사]
{short(draft, 2000)}
"""
        return self.llm(prompt)

    def situation_review(self, draft: str, analysis: dict, plan_json) -> str:
        plan_txt = json.dumps(plan_json, ensure_ascii=False) if isinstance(plan_json, dict) else (plan_json or "")
        prompt = f"""
[Situation Reviewer]
장면 요약("{analysis['scene_summary']}")과의 합치성/이미지 구체성 점검.
결과는 **'수정 지시'**만.

[가이드라인]
{short(plan_txt, 1200)}

[가사]
{short(draft, 2000)}
"""
        return self.llm(prompt)

    # -------- Finalizer --------
    def finalize(self, draft: str, suggestions: str, plan_json) -> str:
        plan_txt = json.dumps(plan_json, ensure_ascii=False) if isinstance(plan_json, dict) else (plan_json or "")
        prompt = f"""
[Finalizer]
아래 '수정 지시'를 반영해 **가사 본문만** 출력하라.
- 설명/머리말/라벨(verse:, chorus:, [Finalizer]) 금지
- 빈 줄로 연/후렴 구분, 힌트 문구 복사 금지
- 반드시 최소 1단어 이상 수정해라. (의미 유지)


[가이드라인]
{short(plan_txt, 1000)}

[수정 지시]
{short(suggestions, 1500)}

[가사 초안]
{short(draft, 2000)}
"""
        return self.llm(prompt)

    # -------- Pipeline --------
    def pipeline(self, analysis: dict, rounds: int = 1, top_k: int = 3,
                 overlap_ngram=3, overlap_thr=0.20, return_trace: bool = False):
        plan = self.plan_guidelines(analysis)
        current, contexts = self.writer(analysis, top_k=top_k)
        trace = {"plan": plan, "contexts": contexts, "rounds": []}

        for _ in range(rounds):
            emo  = self.emotion_review(current, analysis, plan)
            logi = self.logic_review(current, plan)
            situ = self.situation_review(current, analysis, plan)
            merged  = f"[감정]\n{emo}\n\n[형식]\n{logi}\n\n[장면]\n{situ}"
            revised = self.finalize(current, merged, plan)

            if any(overlap_ratio(revised, c, n=overlap_ngram) >= overlap_thr for c in contexts):
                fix = f"""
[Rewrite]
아래 가사가 참조 문맥과 일부 표현이 유사하다.
의미/감정/구조는 유지하되 어휘/문장 표현을 새롭게 바꿔 다시 써라.

[가사]
{short(revised, 2000)}
"""
                revised = self.llm(fix)

            trace["rounds"].append({
                "emotion_review": emo, "logic_review": logi, "situation_review": situ,
                "merged_suggestions": merged, "draft_before": current, "draft_after": revised
            })
            current = revised

        result = current.strip()
        if return_trace:
            trace["final"] = result
            return result, trace
        return result

    # -------- Plain RAG --------
    def plain_rag_generate(self, analysis: dict, top_k=3) -> str:
        query = self.rag.build_query(
            analysis["emotion_label"], analysis["mood_words"],
            analysis["scene_summary"], analysis["genre_hint"], analysis["tempo_hint"]
        )
        ctxs = self.rag.retrieve_contexts(query, top_k=top_k)
        ctx_block = "\n".join(f"- {c}" for c in ctxs)
        prompt = f"""
다음 '표현 힌트'는 영감용. 복사 금지. 출력은 가사 본문만.
구조: verse(4)→chorus(4)→verse(4)→chorus(4)
감정:{analysis['emotion_label']} 장르:{analysis['genre_hint']} 템포:{analysis['tempo_hint']}
장면:{analysis['scene_summary']} 키워드:{", ".join(analysis['mood_words'])}

[표현 힌트]
{short(ctx_block, 1500)}
"""
        return self.llm(prompt)
