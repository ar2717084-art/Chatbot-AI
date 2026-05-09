from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import uuid
import time
import logging

load_dotenv()

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("nova")

# ─── App ───────────────────────────────────────────────────
app = FastAPI(title="Nova AI Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"
MAX_HISTORY = 20      # messages kept per session
SESSION_TTL = 3600    # seconds before idle session is cleared (1 hour)

# ─── Session store ─────────────────────────────────────────
# { session_id: { "history": [...], "last_used": float } }
sessions: dict[str, dict] = {}

# ─── System prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """You are Nova, a smart, friendly, and concise AI assistant.

Capabilities:
- Answer questions clearly and accurately
- Explain concepts in simple language
- Translate text between any languages
- Give creative suggestions and ideas
- Play engaging text-based games
- Write and create high-quality content
- Summarise long text

Tone & Style:
- Warm, helpful, and professional
- Keep answers concise unless the user asks for detail
- Use bullet points or numbered lists when it improves clarity
- Never make up facts — say "I don't know" if unsure
- Respond in the same language the user writes in
"""

# ─── Mode instructions ─────────────────────────────────────
MODE_INSTRUCTIONS = {
    "chat":      lambda msg: msg,
    "info":      lambda msg: f"Explain the following clearly and concisely, using simple language and examples where helpful:\n\n{msg}",
    "translate": lambda msg: f"Detect the language of the following text and translate it to English (or if it's already English, translate to Urdu). Show both the detected language and the translation neatly:\n\n{msg}",
    "suggest":   lambda msg: f"Give 5 practical, creative, and specific suggestions for:\n\n{msg}",
    "game":      lambda msg: f"You are a game master. Play an engaging, interactive text-based game with the user. Current input: {msg}",
    "create":    lambda msg: f"Write high-quality, creative content for the following request. Be original and polished:\n\n{msg}",
    "summarize": lambda msg: f"Summarise the following text in 3-5 bullet points, capturing the key ideas:\n\n{msg}",
    "code":      lambda msg: f"You are an expert programmer. Help with the following coding task. Include explanations:\n\n{msg}",
}

# ─── Schema ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    mode: str = "chat"
    session_id: str | None = None
    language: str = "en"          # reserved for future use

# ─── Helpers ───────────────────────────────────────────────
def clean_sessions():
    """Remove sessions idle longer than SESSION_TTL."""
    now = time.time()
    expired = [sid for sid, data in sessions.items()
               if now - data["last_used"] > SESSION_TTL]
    for sid in expired:
        del sessions[sid]
    if expired:
        log.info(f"Cleaned {len(expired)} expired session(s)")

def get_session(session_id: str) -> dict:
    clean_sessions()
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "last_used": time.time()}
        log.info(f"New session: {session_id[:8]}…")
    sessions[session_id]["last_used"] = time.time()
    return sessions[session_id]

def call_groq(messages: list) -> str:
    """Call Groq API and return the reply text."""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set in .env")

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        timeout=30
    )

    if response.status_code != 200:
        log.error(f"Groq error {response.status_code}: {response.text[:200]}")
        raise RuntimeError(f"Groq API error: {response.status_code}")

    data = response.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError("Unexpected Groq response format")

    return data["choices"][0]["message"]["content"].strip()

# ─── Routes ────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "Nova AI is running 🚀",
        "version": "2.0",
        "model": MODEL,
        "active_sessions": len(sessions)
    }

@app.get("/health")
def health():
    return {"ok": True, "sessions": len(sessions)}

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Allow frontend to reset a conversation."""
    if session_id in sessions:
        del sessions[session_id]
        return {"cleared": True}
    return {"cleared": False}

@app.post("/chat")
async def chat(req: ChatRequest):
    start = time.time()

    try:
        # Resolve session
        session_id = req.session_id or str(uuid.uuid4())
        session = get_session(session_id)
        history: list = session["history"]

        # Validate mode
        mode = req.mode if req.mode in MODE_INSTRUCTIONS else "chat"

        # Build the user message with mode instruction applied
        build = MODE_INSTRUCTIONS[mode]
        user_input = build(req.message.strip())

        # Append to history
        history.append({"role": "user", "content": user_input})

        # Trim history to keep within token budget
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
        session["history"] = history

        # Build full message list for Groq
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history
        ]

        # Call Groq
        reply = call_groq(messages)

        # Save assistant reply to history
        history.append({"role": "assistant", "content": reply})
        session["history"] = history

        elapsed = round((time.time() - start) * 1000)
        log.info(f"[{mode}] session={session_id[:8]}… | {elapsed}ms | {len(reply)} chars")

        return {
            "reply": reply,
            "session_id": session_id,
            "mode": mode,
            "elapsed_ms": elapsed
        }

    except EnvironmentError as e:
        log.error(f"Config error: {e}")
        return JSONResponse(status_code=500, content={
            "reply": "Server configuration error. Please check the API key.",
            "error": str(e)
        })

    except RuntimeError as e:
        log.error(f"Runtime error: {e}")
        return JSONResponse(status_code=502, content={
            "reply": "The AI service is currently unavailable. Please try again.",
            "error": str(e)
        })

    except requests.exceptions.Timeout:
        log.error("Groq request timed out")
        return JSONResponse(status_code=504, content={
            "reply": "The request timed out. Please try again.",
        })

    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={
            "reply": "Something went wrong on the server.",
            "error": str(e)
        })  