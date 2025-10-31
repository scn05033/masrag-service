# app/main.py
import os, time, logging
import httpx
import traceback
from fastapi import Query
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware

from .rag import RAGStore
from .mas import MASRAG
from .image_emotion import ImageEmotion
from dotenv import load_dotenv
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

logger = logging.getLogger("masrag")
logger.setLevel(logging.INFO)

TRACE_LOG = os.getenv("TRACE_LOG", "false").lower() == "true"

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path, override=True)
print("[ENV] AUTO_AUDIO:", os.getenv("AUTO_AUDIO"))
print("[ENV] SUNO_API_BASE:", os.getenv("SUNO_API_BASE"))
print("[ENV] SUNO_API_KEY len:", len(os.getenv("SUNO_API_KEY") or ""))
SUNO_API_BASE = os.getenv("SUNO_API_BASE")
SUNO_API_KEY  = os.getenv("SUNO_API_KEY")

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

app = FastAPI(title="MAS-RAG Lyrics Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시 특정 도메인으로 제한하세요
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 핸들: 서버 생명주기 동안 유지
rag_store = RAGStore()
masrag = None
imgemo = None

@app.on_event("startup")
def startup():
    global masrag, imgemo
    rag_store.load_or_build(force_rebuild=False)  # 최초 1회 로드/구축
    masrag = MASRAG(rag_store)
    imgemo = ImageEmotion()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/admin/reindex")
def reindex(background_tasks: BackgroundTasks):
    background_tasks.add_task(rag_store.load_or_build, True)
    return {"status": "reindex started"}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        mime = file.content_type or "image/jpeg"
        analysis = imgemo.analyze(img_bytes, mime=mime)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(500, f"analyze_image failed: {e}")

@app.post("/generate-lyrics")
async def generate_lyrics(
    request: Request,
    debug: bool = Query(False, description="과정(trace)까지 응답"),
    rounds: int = Query(1, ge=1, le=3, description="MAS 라운드 수"),
    top_k: int = Query(3, ge=1, le=10, description="RAG 검색 개수"),
    use_mas: bool = Query(True, description="True면 MAS-RAG, False면 단일 RAG")
):
    """
    가사 생성 엔드포인트.
    - 기본은 MAS-RAG 파이프라인 (use_mas=True)
    - debug=true 이면 trace(plan/contexts/reviews/drafts...)까지 반환
    - 쿼리스트링과 JSON 바디 모두 지원(바디 값이 있으면 바디가 우선)
    """
    try:
        # JSON 바디 읽기 (없으면 {} 처리)
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        # 필수 입력: analysis
        analysis = payload.get("analysis")
        if analysis is None:
            raise HTTPException(status_code=400, detail="`analysis` 필드가 필요합니다 (이미지 감정 분석 JSON).")

        # 바디가 쿼리 파라미터를 덮어쓰도록 허용
        if "use_mas" in payload:
            use_mas = bool(payload["use_mas"])
        if "rounds" in payload:
            rounds = int(payload["rounds"])
        if "top_k" in payload:
            top_k = int(payload["top_k"])
        if "debug" in payload:
            debug = bool(payload["debug"])

        # 파이프라인 실행
        if debug:
            if use_mas:
                lyrics, trace = masrag.pipeline(
                    analysis, rounds=rounds, top_k=top_k, return_trace=True
                )
                mode = "MAS-RAG"
                pretty = {
                    "plan": trace.get("plan"),
                    "contexts": trace.get("contexts"),
                    "rounds": []
                }
                for r in trace.get("rounds", []):
                    pretty["rounds"].append({
                        "emotion_review": (r.get("emotion_review") or "").strip(),
                        "logic_review": (r.get("logic_review") or "").strip(),
                        "situation_review": (r.get("situation_review") or "").strip(),
                        "draft_before": (r.get("draft_before") or "").strip()[:500],
                        "draft_after": (r.get("draft_after") or "").strip()[:500],
                    })
                pretty["final"] = (trace.get("final") or lyrics or "").strip()

                return {
                    "ok": True,
                    "mode": "MAS-RAG",
                    "params": {"use_mas": use_mas, "rounds": rounds, "top_k": top_k},
                    "lyrics": lyrics,
                    "trace": pretty
                }
            else:
                lyrics = masrag.plain_rag_generate(analysis, top_k=top_k)
                trace = {
                    "plan": None,
                    "contexts": rag_store.retrieve_contexts(  
                        rag_store.build_query(
                            analysis.get("emotion_label", ""),
                            analysis.get("mood_words", []),
                            analysis.get("scene_summary", ""),
                            analysis.get("genre_hint", ""),
                            analysis.get("tempo_hint", "")
                        ),
                        top_k=top_k
                    ),
                    "rounds": [],
                    "final": lyrics,
                    "mode": "plain_rag"
                }
                mode = "RAG"
            return {
                "ok": True,
                "mode": mode,
                "params": {"use_mas": use_mas, "rounds": rounds, "top_k": top_k},
                "lyrics": lyrics,
                "trace": trace
            }
        else:
            if use_mas:
                lyrics = masrag.pipeline(analysis, rounds=rounds, top_k=top_k, return_trace=False)
                mode = "MAS-RAG"
            else:
                lyrics = masrag.plain_rag_generate(analysis, top_k=top_k)
                mode = "RAG"
            return {
                "ok": True,
                "mode": mode,
                "params": {"use_mas": use_mas, "rounds": rounds, "top_k": top_k},
                "lyrics": lyrics
            }

    except HTTPException:
        raise
    except Exception as e:
        # 내부 예외는 500으로 래핑
        raise HTTPException(status_code=500, detail=f"generate_lyrics failed: {e}")
    

@app.get("/", response_class=HTMLResponse)
def home():
    return open("static/index.html", "r", encoding="utf-8").read()
SUNO_API_BASE = os.getenv("SUNO_API_BASE")  
SUNO_API_KEY  = os.getenv("SUNO_API_KEY")

def _suno_headers():
    if not SUNO_API_KEY or not SUNO_API_BASE:
        raise RuntimeError("SUNO_API_BASE / SUNO_API_KEY not set")
    return {"Authorization": f"Bearer {SUNO_API_KEY}"}

@app.post("/generate-audio")
async def generate_audio(payload: dict):
    """
    입력: { "lyrics": "...", "style": "ballad", "title": "비 오는 골목", "mv": "v5" }
    출력: { "task_id": "xxxx" }
    """
    try:
        lyrics = payload.get("lyrics", "")
        title  = payload.get("title", "Untitled")
        style  = payload.get("style", "pop")
        model  = payload.get("mv", "v5")  
        url = f"{SUNO_API_BASE}/api/generate"
        req = {
            "prompt": f"title:{title}\nstyle:{style}\nlyrics:{lyrics}",
            "model": model,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=req, headers=_suno_headers())
            resp.raise_for_status()
            data = resp.json()
        task_id = data.get("data", {}).get("taskId") or data.get("taskId")
        if not task_id:
            raise RuntimeError(f"no taskId in response: {data}")
        return {"task_id": task_id}
    except Exception as e:
        print("[/generate-audio] ERROR", repr(e))
        raise HTTPException(500, f"generate_audio failed: {e}")

@app.get("/audio-status")
async def audio_status(task_id: str = Query(..., description="music task id")):
    """
    입력: /audio-status?task_id=xxxx
    출력 예: { "status":"done", "songs":[{"audio_url":"...", "title":"..."}] }
    """
    try:
        url = f"{SUNO_API_BASE}/api/v1/generate/record-info"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, params={"taskId": task_id}, headers=_suno_headers())
            body = r.text
            r.raise_for_status()
            data = r.json()

       
        row = (data.get("data") or {})
        resp = row.get("response") or {}
        suno = (resp.get("sunoData") or [])
        songs = []
        done = False
        for it in suno:
            audio = it.get("audioUrl") or it.get("streamAudioUrl")
            title = it.get("title") or ""
            status = row.get("status") or ""
            if audio: done = True
            songs.append({"status": status, "audio_url": audio, "title": title})

        return {"status": "done" if done else (row.get("status") or "pending"), "songs": songs}
    except httpx.HTTPStatusError as he:
        return {"status": "error", "error": f"HTTP {he.response.status_code}", "detail": he.response.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/create-song")
async def create_song(
    file: UploadFile = File(...),
    use_mas: bool = Form(True),
    rounds: int = Form(1),
    top_k: int = Form(3),
    debug: bool = Form(False), 
    
    
):
    """
    이미지 업로드 → 감정/장면 분석(Gemini) → (MAS)RAG 가사 생성
    AUTO_AUDIO=true 이고 Suno API 설정이 있으면 곧바로 '음악 생성' 작업을 트리거하여 task_id 반환
    응답: { analysis, lyrics, task_id }
    """
    try:
        # 1) 업로드 이미지 검증
        img_bytes = await file.read()
        mime = file.content_type or "image/jpeg"
        if not img_bytes or len(img_bytes) < 1024:
            raise ValueError("빈 파일이거나 너무 작은 이미지입니다.")
        if not mime.startswith("image/"):
            raise ValueError(f"이미지 MIME 아님: {mime}")

        # 2) 이미지 감정/장면 분석 (Gemini)
        analysis = imgemo.analyze(img_bytes, mime=mime)

        # 3) (MAS-)RAG 가사 생성
        if use_mas:
            if debug:
                lyrics, trace = masrag.pipeline(analysis, rounds=rounds, top_k=top_k, return_trace=True)
            else:
                lyrics = masrag.pipeline(analysis, rounds=rounds, top_k=top_k, return_trace=False)
                trace = None
        else:
            lyrics = masrag.plain_rag_generate(analysis, top_k=top_k)
            if debug:
            # 단일 RAG일 때도 컨텍스트는 보여주자
                q = rag_store.build_query(
                analysis.get("emotion_label",""),
                analysis.get("mood_words",[]),
                analysis.get("scene_summary",""),
                analysis.get("genre_hint",""),
                analysis.get("tempo_hint",""),
            )
                ctxs = rag_store.retrieve_contexts(q, top_k=top_k)
                trace = {"plan": None, "contexts": ctxs, "rounds": [], "final": lyrics, "mode": "RAG"}
            else:
                trace = None

        # 4) (옵션) Suno API로 자동 음악 생성 트리거
        task_id = None
        auto_audio = (os.getenv("AUTO_AUDIO", "false").lower() == "true")
        suno_ready = bool(SUNO_API_BASE and SUNO_API_KEY)

        if auto_audio and suno_ready:
            try:
                
                payload = {
                    "customMode": False,          # 간단 모드
                    "instrumental": False,        # 보컬 포함
                    "model": "V5",                # 모델 버전 (문서에 맞춰 수정 가능)
                    "prompt": lyrics,             # 간단 모드에서는 prompt만 필수
                    "callBackUrl": "https://example.com/callback"  # TODO: 필요 시 실제 콜백 URL로 변경
                }
                url = f"{SUNO_API_BASE.rstrip('/')}/api/v1/generate"
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.post(url, json=payload, headers=_suno_headers())
                    body = r.text
                    r.raise_for_status()
                    data = r.json()
                # 보편적 응답: {"code":200, "data":{"taskId":"..."} ...}
                task_id = (data.get("data") or {}).get("taskId") or data.get("taskId")
                if not task_id:
                    raise RuntimeError(f"no taskId in response: {body[:200]}")
            except Exception as ae:
                # 음악 생성은 실패하더라도 가사 생성 결과는 반환
                print("[AUDIO AUTO] failed:", ae)

        elif auto_audio and not suno_ready:
            # AUTO_AUDIO=true 이지만 BASE/KEY가 없을 때는 경고만 남기고 계속
            print("[AUDIO AUTO] skipped: SUNO_API_BASE / SUNO_API_KEY not set")

        # 5) 결과 반환
        return {"analysis": analysis, "lyrics": lyrics, "task_id": task_id, "trace": trace}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"create_song failed: {type(e).__name__}: {e}")
app.mount("/static", StaticFiles(directory="static"), name="static")