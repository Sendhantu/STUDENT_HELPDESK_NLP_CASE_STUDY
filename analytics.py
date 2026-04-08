"""
analytics.py — SQLite-based query analytics logger
Logs every chat query with: timestamp, message, source, confidence, intent, language.
Provides helper methods to query top failed intents and usage trends.
"""

import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path("analytics.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the analytics table if it doesn't exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                message     TEXT    NOT NULL,
                source      TEXT    NOT NULL,   -- 'tfidf' | 'groq' | 'system'
                confidence  REAL    DEFAULT 0,
                intent      TEXT,
                language    TEXT    DEFAULT 'en',
                session_id  TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source    ON query_log(source);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_intent    ON query_log(intent);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON query_log(timestamp);
        """)
        conn.commit()
    log.info("Analytics DB initialised at %s", DB_PATH)


def log_query(
    message: str,
    source: str,
    confidence: float = 0.0,
    intent: str | None = None,
    language: str = "en",
    session_id: str | None = None,
) -> None:
    """Insert a single query record."""
    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO query_log
                    (timestamp, message, source, confidence, intent, language, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    message[:1000],     # cap to avoid runaway storage
                    source,
                    round(confidence, 4),
                    intent,
                    language,
                    session_id,
                ),
            )
            conn.commit()
    except Exception as e:
        log.error("Failed to log query to analytics DB: %s", e)


# ── Query helpers ──────────────────────────────────────────────────────────

def get_top_failed_queries(limit: int = 20) -> list[dict]:
    """
    Returns queries that fell through to Groq (TF-IDF couldn't answer).
    Useful for identifying gaps in the knowledge base.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT message, COUNT(*) as count, AVG(confidence) as avg_conf
            FROM query_log
            WHERE source = 'groq'
            GROUP BY LOWER(message)
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_intent_distribution(days: int = 30) -> list[dict]:
    """Returns intent hit counts over the last N days."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT intent, source, COUNT(*) as count
            FROM query_log
            WHERE timestamp >= ? AND intent IS NOT NULL
            GROUP BY intent, source
            ORDER BY count DESC
            """,
            (since,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_language_distribution(days: int = 30) -> list[dict]:
    """Returns query language breakdown."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT language, COUNT(*) as count
            FROM query_log
            WHERE timestamp >= ?
            GROUP BY language
            ORDER BY count DESC
            """,
            (since,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_summary(days: int = 7) -> dict:
    """Returns a summary dashboard dict for the /analytics endpoint."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _get_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE timestamp >= ?", (since,)
        ).fetchone()[0]

        tfidf_count = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE source='tfidf' AND timestamp >= ?",
            (since,)
        ).fetchone()[0]

        groq_count = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE source='groq' AND timestamp >= ?",
            (since,)
        ).fetchone()[0]

        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM query_log WHERE source='tfidf' AND timestamp >= ?",
            (since,)
        ).fetchone()[0] or 0

    return {
        "period_days":      days,
        "total_queries":    total,
        "tfidf_answered":   tfidf_count,
        "groq_fallback":    groq_count,
        "tfidf_rate_%":     round((tfidf_count / total * 100) if total else 0, 1),
        "avg_tfidf_conf_%": round(avg_conf * 100, 1),
        "top_failed":       get_top_failed_queries(10),
        "intent_dist":      get_intent_distribution(days),
        "language_dist":    get_language_distribution(days),
    }
