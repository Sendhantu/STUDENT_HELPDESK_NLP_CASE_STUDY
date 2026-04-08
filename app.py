import os
import re
import hashlib
import logging
from datetime import datetime, timezone
from collections import Counter

import requests
from flask import Flask, request, jsonify, render_template, session

try:
    from flask_caching import Cache
except ImportError:
    class Cache:  # type: ignore[override]
        def __init__(self, app=None):
            self._store = {}

        def get(self, key):
            return self._store.get(key)

        def set(self, key, value, timeout=None):
            self._store[key] = value

        def clear(self):
            self._store.clear()

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except ImportError:
    def get_remote_address():
        return request.remote_addr or "local"

    class Limiter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def limit(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

try:
    from flask_session import Session
except ImportError:
    class Session:  # type: ignore[override]
        def __init__(self, app=None):
            if app is not None:
                self.init_app(app)

        def init_app(self, app):
            return app

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

from knowledge_base import (
    QA_PAIRS,
    find_by_intent,
    get_debug_snapshot,
    get_step_answer,
    kb_summary,
    reload_from_disk,
    search_pairs,
)
from nlp_engine import TFIDFChatbot
from analytics import init_db, log_query, get_summary

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

TANGLISH_MARKERS = {
    "fees", "padikanum", "exam", "attend", "percentage", "result",
    "sollu", "enna", "epdi", "eppadi", "yenna", "yellam", "theriyuma",
    "sema", "super", "vera", "level", "konjam", "romba", "oru",
    "illa", "illai", "varum", "varuma", "panrom", "panren",
    "paakanum", "paaru", "teriuma", "sollunga", "solluvanga",
    "ungaluku", "naan", "naanga", "avanga", "ivan", "avan",
}


# ── Config ─────────────────────────────────────────────────────────────────
class Config:
    # ── Groq ──
    GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
    GROQ_API_URL     = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL       = "llama3-8b-8192"
    GROQ_TIMEOUT     = 10
    GROQ_MAX_TOKENS  = 250
    GROQ_TEMPERATURE = 0.4

    # ── NLP ──
    TFIDF_THRESHOLD  = 0.25

    # ── Session (server-side filesystem) ──
    SECRET_KEY         = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")
    SESSION_TYPE       = "filesystem"
    SESSION_FILE_DIR   = "./flask_sessions"
    SESSION_PERMANENT  = False
    MAX_HISTORY_TURNS  = 6          # user+bot pairs kept in context

    # ── Cache ──
    CACHE_TYPE         = "SimpleCache"   # swap to "RedisCache" in production
    CACHE_DEFAULT_TIMEOUT = 300          # 5 minutes

    # ── Input ──
    MAX_INPUT_LENGTH   = 500

    # ── Rate limiting ──
    RATE_LIMIT_DEFAULT = "60 per hour"
    RATE_LIMIT_CHAT    = "20 per minute"


# ── App & extensions ───────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

# Server-side sessions
os.makedirs(Config.SESSION_FILE_DIR, exist_ok=True)
Session(app)

# Response cache
cache = Cache(app)

# Rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[Config.RATE_LIMIT_DEFAULT],
    storage_uri="memory://",
)

# NLP engine
chatbot = TFIDFChatbot(QA_PAIRS)

# Analytics DB
init_db()


# ── System prompts ─────────────────────────────────────────────────────────
SYSTEM_PROMPT_EN = """
You are a Student Helpdesk Assistant for SRM Institute of Science and Technology.
Answer ONLY college-related questions. Be concise (≤4 sentences) and helpful.
Key facts:
- Semester fee: ₹1,10,000 (B.Tech CSE)
- Exams: Internal (3 cycles, 40 marks) + End Semester (60 marks, Nov/April)
- Classes: 8 AM–5 PM, Mon–Sat; Lunch: 12:50–1:40 PM
- Portal: https://academia.srmist.edu.in
- Helpdesk email: helpdesk@srmist.edu.in
If unsure, direct the student to helpdesk@srmist.edu.in or the admin office.
Never answer non-college topics. Politely redirect if off-topic.
""".strip()

SYSTEM_PROMPT_TANGLISH = """
You are a Student Helpdesk Assistant for SRM Institute of Science and Technology.
The student is writing in Tanglish (Tamil mixed with English). Understand their question
and reply in simple, clear English (do NOT reply in Tamil or Tanglish).
Answer ONLY college-related questions. Be concise (≤4 sentences).
Key facts:
- Semester fee: ₹1,10,000 (B.Tech CSE)
- Exams: Internal (3 cycles, 40 marks) + End Semester (60 marks, Nov/April)
- Classes: 8 AM–5 PM, Mon–Sat; Lunch: 12:50–1:40 PM
- Portal: https://academia.srmist.edu.in
- Helpdesk email: helpdesk@srmist.edu.in
If unsure, direct the student to helpdesk@srmist.edu.in.
""".strip()


# ── Helpers ────────────────────────────────────────────────────────────────
def sanitize_input(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    """
    Returns a language code: 'en', 'ta', or 'tanglish'.
    Tanglish = English-script text containing Tamil vocabulary.
    """
    # Check for Tanglish markers first (before langdetect)
    lower = text.lower()
    tokens = set(re.findall(r"\b\w+\b", lower))
    if tokens & TANGLISH_MARKERS:
        return "tanglish"

    if not LANGDETECT_AVAILABLE:
        return "en"

    try:
        lang = detect(text)
        # 'ta' = Tamil script; treat as Tamil
        if lang == "ta":
            return "ta"
        return "en"
    except LangDetectException:
        return "en"


def get_cache_key(message: str) -> str:
    """Stable cache key from the normalised message."""
    normalised = message.lower().strip()
    return "tfidf:" + hashlib.md5(normalised.encode()).hexdigest()


def get_session_history() -> list[dict]:
    history = session.get("history", [])
    normalised: list[dict] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        normalised.append({
            "role": entry.get("role", "assistant"),
            "content": entry.get("content", ""),
            "meta": entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {},
        })
    return normalised


def get_model_history() -> list[dict]:
    """Return history in the minimal format expected by the LLM API."""
    model_history = []
    for entry in get_session_history():
        role = entry.get("role", "assistant")
        if role not in {"user", "assistant", "system"}:
            role = "assistant"
        content = entry.get("content", "")
        if content:
            model_history.append({"role": role, "content": content})
    return model_history


def append_to_history(role: str, content: str, meta: dict | None = None) -> None:
    history = get_session_history()
    history.append({
        "role": role,
        "content": content,
        "meta": meta or {},
    })
    cap = Config.MAX_HISTORY_TURNS * 2
    if len(history) > cap:
        history = history[-cap:]
    session["history"] = history


def get_session_id() -> str:
    """Return or create a stable session identifier."""
    if "sid" not in session:
        import uuid
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def ask_groq(user_message: str, language: str = "en") -> str:
    """Call Groq with the appropriate system prompt based on detected language."""
    if not Config.GROQ_API_KEY:
        log.error("GROQ_API_KEY not set.")
        return "LLM fallback unavailable. Please contact helpdesk@srmist.edu.in."

    system_prompt = (
        SYSTEM_PROMPT_TANGLISH
        if language in ("tanglish", "ta")
        else SYSTEM_PROMPT_EN
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_model_history())
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {Config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":       Config.GROQ_MODEL,
        "messages":    messages,
        "max_tokens":  Config.GROQ_MAX_TOKENS,
        "temperature": Config.GROQ_TEMPERATURE,
    }

    try:
        res = requests.post(
            Config.GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=Config.GROQ_TIMEOUT,
        )
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"].strip()
        log.info("Groq OK (lang=%s tokens=%s)",
                 language,
                 res.json().get("usage", {}).get("total_tokens", "?"))
        return reply

    except requests.exceptions.Timeout:
        log.warning("Groq timeout after %ss", Config.GROQ_TIMEOUT)
        return "Response is taking too long. Please try again or contact helpdesk@srmist.edu.in."

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        log.error("Groq HTTP %s", status)
        if status == 429:
            return "AI engine is busy. Please wait a moment and try again."
        return "Could not process your request. Please contact helpdesk@srmist.edu.in."

    except Exception as e:
        log.exception("Groq unexpected error: %s", e)
        return "An unexpected error occurred. Please contact helpdesk@srmist.edu.in."


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
@limiter.limit(Config.RATE_LIMIT_CHAT)
def chat():
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload."}), 400

    raw = data.get("message", "")
    if not isinstance(raw, str):
        return jsonify({"error": "Message must be a string."}), 400

    user_message = sanitize_input(raw)

    if not user_message:
        return jsonify({"reply": "Please type a question!",
                        "source": "system", "confidence": 0})

    if len(user_message) > Config.MAX_INPUT_LENGTH:
        return jsonify({
            "reply": f"Please keep your message under {Config.MAX_INPUT_LENGTH} characters.",
            "source": "system", "confidence": 0,
        })

    # ── Language detection ──────────────────────────────────────────────
    language = detect_language(user_message)
    log.info("Query (lang=%s): %.80s", language, user_message)

    sid = get_session_id()

    # ── Step 1: TF-IDF with caching ────────────────────────────────────
    cache_key = get_cache_key(user_message)
    cached = cache.get(cache_key)

    if cached:
        log.info("Cache HIT for query")
        answer, score, matched_q, intent = cached
    else:
        answer, score, matched_q, intent = chatbot.get_response(user_message)
        if answer and score >= Config.TFIDF_THRESHOLD:
            cache.set(cache_key, (answer, score, matched_q, intent))

    if answer and score >= Config.TFIDF_THRESHOLD:
        pair = find_by_intent(intent) or {}
        assistant_meta = {
            "source": "tfidf",
            "confidence": round(score * 100, 1),
            "intent": intent,
            "language": language,
            "steps": get_step_answer(intent) or [],
            "office": pair.get("source", ""),
            "tags": pair.get("tags", []),
            "matched": matched_q,
        }
        append_to_history("user", user_message, {"language": language})
        append_to_history("assistant", answer, assistant_meta)
        log_query(user_message, "tfidf", score, intent, language, sid)
        log.info("TF-IDF matched (score=%.2f intent=%s)", score, intent)
        return jsonify({
            "reply":      answer,
            "source":     "tfidf",
            "confidence": round(score * 100, 1),
            "matched":    matched_q,
            "intent":     intent,
            "language":   language,
            "steps":      assistant_meta["steps"],
            "office":     pair.get("source", ""),
            "tags":       pair.get("tags", []),
        })

    # ── Step 2: Groq LLM fallback ──────────────────────────────────────
    log.info("TF-IDF %.2f < threshold → Groq (lang=%s)", score, language)
    groq_reply = ask_groq(user_message, language)
    suggestions = [
        hit["question"]
        for hit in chatbot.top_k(user_message, k=3)
        if hit.get("score", 0) >= 0.08
    ]
    append_to_history("user", user_message, {"language": language})
    append_to_history("assistant", groq_reply, {
        "source": "groq",
        "confidence": round(score * 100, 1),
        "intent": "llm_fallback",
        "language": language,
        "suggestions": suggestions,
    })
    log_query(user_message, "groq", score, "llm_fallback", language, sid)

    return jsonify({
        "reply":      groq_reply,
        "source":     "groq",
        "confidence": round(score * 100, 1),
        "intent":     "llm_fallback",
        "language":   language,
        "suggestions": suggestions,
    })


@app.route("/history", methods=["GET"])
def history():
    return jsonify({"history": get_session_history()})


@app.route("/history", methods=["DELETE"])
def clear_history():
    session.pop("history", None)
    session.pop("sid", None)
    log.info("Session cleared.")
    return jsonify({"message": "Chat history cleared."})


@app.route("/stats", methods=["GET"])
def stats():
    summary = kb_summary()
    intent_counts = Counter(p["intent"] for p in QA_PAIRS)
    return jsonify({
        "total_qa_pairs":    summary["total_pairs"],
        "intents":           dict(intent_counts),
        "domains":           summary["domains"],
        "with_steps":        summary["with_steps"],
        "unique_tags":       summary["unique_tags"],
        "unique_sources":    summary["unique_sources"],
        "featured_questions": summary["featured_questions"],
        "featured_items":    summary["featured_items"],
        "stale_count":       summary["stale_count"],
        "kb_meta":           summary["meta"],
        "tfidf_threshold":   Config.TFIDF_THRESHOLD,
        "groq_model":        Config.GROQ_MODEL,
    })


@app.route("/analytics", methods=["GET"])
def analytics():
    """7-day usage summary: TF-IDF hit rate, top failures, intent dist, languages."""
    days = int(request.args.get("days", 7))
    return jsonify(get_summary(days))


@app.route("/admin/debug", methods=["GET"])
def admin_debug():
    """Combined KB and analytics snapshot for local debugging/admin use."""
    days = int(request.args.get("days", 7))
    limit = int(request.args.get("limit", 8))
    query = sanitize_input(request.args.get("q", ""))

    payload = {
        "knowledge_base": get_debug_snapshot(limit=limit),
        "analytics": get_summary(days),
    }
    if query:
        payload["search_results"] = [
            {
                "question": pair.get("question", ""),
                "intent": pair.get("intent", ""),
                "source": pair.get("source", "Unknown"),
                "tags": pair.get("tags", []),
            }
            for pair in search_pairs(query, limit=limit)
        ]
    return jsonify(payload)


@app.route("/admin/reload", methods=["POST"])
def admin_reload():
    """Hot-reload the knowledge base without restarting the server."""
    try:
        fresh_pairs = reload_from_disk()
        chatbot.reload(fresh_pairs)
        cache.clear()
        log.info("Knowledge base reloaded: %d pairs", len(fresh_pairs))
        return jsonify({
            "message": f"Reloaded {len(fresh_pairs)} QA pairs. Cache cleared."
        })
    except Exception as e:
        log.exception("Reload failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "timestamp":     datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kb_size":       len(QA_PAIRS),
        "groq_key_set":  bool(Config.GROQ_API_KEY),
        "langdetect":    LANGDETECT_AVAILABLE,
        "cache_type":    Config.CACHE_TYPE,
        "session_type":  Config.SESSION_TYPE,
    })


# ── Error handlers ─────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({
        "error": "Too many requests. Please slow down.",
        "retry_after": str(e.description),
    }), 429

@app.errorhandler(500)
def internal_error(e):
    log.exception("Unhandled 500: %s", e)
    return jsonify({"error": "Internal server error."}), 500


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not Config.GROQ_API_KEY:
        log.warning("⚠️  GROQ_API_KEY not set — LLM fallback disabled.")
    if Config.SECRET_KEY == "change-me-in-production":
        log.warning("⚠️  Using default SECRET_KEY — set FLASK_SECRET_KEY in production.")
    if not LANGDETECT_AVAILABLE:
        log.warning("⚠️  langdetect not installed — language detection disabled.")
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
