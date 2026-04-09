import os, re, hashlib, logging
from datetime import datetime, timezone
from collections import Counter
import requests
from flask import Flask, request, jsonify, render_template, session
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_session import Session
from langdetect import detect, LangDetectException
from knowledge_base import QA_PAIRS, find_by_intent, get_step_answer, kb_summary, reload_from_disk, search_pairs
from nlp_engine import TFIDFChatbot
from analytics import init_db, log_query, get_summary

log = logging.getLogger(__name__)

TANGLISH_MARKERS = {
    "fees", "padikanum", "exam", "sollu", "enna", "epdi", "sema",
    "romba", "konjam", "illa", "varum", "panrom", "paakanum",
}

class Config:
    GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
    GROQ_API_URL     = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL       = "llama3-8b-8192"
    TFIDF_THRESHOLD  = 0.25
    SECRET_KEY       = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")
    SESSION_TYPE     = "filesystem"
    MAX_HISTORY_TURNS = 6
    CACHE_TYPE       = "SimpleCache"
    MAX_INPUT_LENGTH = 500
    RATE_LIMIT_CHAT  = "20 per minute"

app = Flask(__name__)
app.config.from_object(Config)
Session(app); cache = Cache(app)
limiter = Limiter(key_func=get_remote_address, app=app, storage_uri="memory://")
chatbot = TFIDFChatbot(QA_PAIRS)
init_db()

SYSTEM_PROMPT_EN = """You are a Student Helpdesk Assistant for SRM IST.
Answer ONLY college-related questions. Be concise (≤4 sentences).
Key facts: Fee ₹1,10,000 | Exams: Internal (40) + End Sem (60) | Portal: academia.srmist.edu.in"""

def detect_language(text):
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    if tokens & TANGLISH_MARKERS: return "tanglish"
    try: return detect(text)
    except: return "en"

def append_to_history(role, content, meta=None):
    history = session.get("history", [])
    history.append({"role": role, "content": content, "meta": meta or {}})
    session["history"] = history[-Config.MAX_HISTORY_TURNS * 2:]

def ask_groq(user_message, language="en"):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_EN}]
    messages += session.get("history", [])
    messages.append({"role": "user", "content": user_message})
    res = requests.post(Config.GROQ_API_URL,
        headers={"Authorization": f"Bearer {Config.GROQ_API_KEY}"},
        json={"model": Config.GROQ_MODEL, "messages": messages,
              "max_tokens": 250, "temperature": 0.4}, timeout=10)
    return res.json()["choices"][0]["message"]["content"].strip()

@app.route("/")
def index(): return render_template("index.html")

@app.route("/chat", methods=["POST"])
@limiter.limit(Config.RATE_LIMIT_CHAT)
def chat():
    user_message = re.sub(r"<[^>]+>", "", request.get_json().get("message", "")).strip()
    if not user_message: return jsonify({"reply": "Please type a question!"})
    language = detect_language(user_message)
    cache_key = "tfidf:" + hashlib.md5(user_message.lower().encode()).hexdigest()
    cached = cache.get(cache_key)
    answer, score, matched_q, intent = cached if cached else chatbot.get_response(user_message)
    if answer and score >= Config.TFIDF_THRESHOLD:
        if not cached: cache.set(cache_key, (answer, score, matched_q, intent))
        append_to_history("user", user_message)
        append_to_history("assistant", answer)
        log_query(user_message, "tfidf", score, intent, language)
        return jsonify({"reply": answer, "source": "tfidf",
                        "confidence": round(score * 100, 1), "intent": intent})
    groq_reply = ask_groq(user_message, language)
    append_to_history("user", user_message)
    append_to_history("assistant", groq_reply)
    log_query(user_message, "groq", score, "llm_fallback", language)
    return jsonify({"reply": groq_reply, "source": "groq"})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "kb_size": len(QA_PAIRS),
                    "groq_key_set": bool(Config.GROQ_API_KEY)})

@app.route("/admin/reload", methods=["POST"])
def admin_reload():
    fresh = reload_from_disk(); chatbot.reload(fresh); cache.clear()
    return jsonify({"message": f"Reloaded {len(fresh)} QA pairs."})

if __name__ == "__main__":
    app.run(debug=False)
