# app/image_emotion.py
import os, json
from google import genai

 # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMOTIONS = os.getenv("EMOTIONS", "calm,happy,lonely,nostalgic,tense,majestic").split(",")
GENRES   = os.getenv("GENRES",   "ballad,pop,rock,ambient,orchestral").split(",")
TEMPOS   = os.getenv("TEMPOS",   "slow,mid,fast").split(",")

class ImageEmotion:
    def __init__(self):
        key = os.getenv("GEMINI_API_KEY")           # ← 여기서 읽기
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=key)

    def analyze(self, image_bytes: bytes, mime: str = "image/jpeg") -> dict:
        system = (
            "You analyze a landscape image and return strict JSON with double quotes only:\n"
            "{\n"
            f'  "emotion_label": "<one of: {", ".join(EMOTIONS)}>",\n'
            '  "mood_words": ["<3~5 short words in Korean>"],\n'
            '  "scene_summary": "<<=40 Korean words>",\n'
            f'  "tempo_hint": "<one of: {", ".join(TEMPOS)}>",\n'
            f'  "genre_hint": "<one of: {", ".join(GENRES)}>"\n'
            "}\n"
            "Return ONLY the JSON."
        )
        resp = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role":"user","parts":[
                {"text": system},
                {"inline_data": {"mime_type": mime, "data": image_bytes}},
                {"text": "Analyze and return ONLY the JSON."}
            ]}],
            config={"response_mime_type":"application/json"}
        )
        
        return json.loads(resp.text)
