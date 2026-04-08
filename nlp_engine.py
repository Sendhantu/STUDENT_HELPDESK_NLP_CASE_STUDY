import re
import string
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "of", "to", "and", "or",
    "for", "on", "at", "are", "was", "what", "how", "when", "where",
    "who", "which", "i", "my", "me", "do", "can", "you", "please",
    "tell", "let", "know", "about", "will", "be", "this", "that",
}

SYNONYMS: dict[str, str] = {
    "tuition": "fee",
    "cost": "fee",
    "price": "fee",
    "charges": "fee",
    "amount": "fee",
    "pay": "fee",
    "payment": "fee",
    "hostel": "accommodation",
    "dorm": "accommodation",
    "dormitory": "accommodation",
    "room": "accommodation",
    "quiz": "internal",
    "cat": "internal",
    "cycle test": "internal",
    "unit test": "internal",
    "semester exam": "end semester",
    "final exam": "end semester",
    "end sem": "end semester",
    "timetable": "schedule",
    "timing": "schedule",
    "class time": "schedule",
    "login": "portal",
    "academia": "portal",
    "website": "portal",
    "mail": "email",
    "contact": "helpdesk",
    "support": "helpdesk",
    "financial aid": "scholarship",
    "stipend": "scholarship",
    "backlog": "arrear",
    "failed": "arrear",
    "reexam": "arrear",
    "grade": "marks",
    "result": "marks",
    "cgpa": "marks",
    "gpa": "marks",
}

TOKEN_RE = re.compile(r"\b\w+\b")


class Preprocessor:
    def normalise(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b\d+\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        normalised = self.normalise(text)
        tokens = [
            SYNONYMS.get(token, token)
            for token in normalised.split()
            if token not in STOPWORDS and len(token) > 1
        ]
        joined = " ".join(tokens)
        for phrase, replacement in SYNONYMS.items():
            if " " in phrase:
                joined = joined.replace(phrase, replacement)
        return [token for token in joined.split() if token]

    def __call__(self, text: str) -> str:
        return " ".join(self.tokenize(text))


@dataclass
class ChatbotResponse:
    answer: Optional[str]
    score: float
    matched_q: Optional[str]
    intent: Optional[str]

    def as_tuple(self):
        return self.answer, self.score, self.matched_q, self.intent


class TFIDFChatbot:
    def __init__(self, qa_pairs: list[dict], threshold: float = 0.25):
        self.qa_pairs = qa_pairs
        self.threshold = threshold
        self.preprocessor = Preprocessor()
        self._build_index()

    def _prepare_word_doc(self, pair: dict) -> str:
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        keywords = " ".join(pair.get("keywords", []))
        tags = " ".join(pair.get("tags", []))
        aliases = " ".join(pair.get("aliases", []))
        related = " ".join(pair.get("related_questions", []))
        steps = " ".join(pair.get("answer_steps", []))
        intent = pair.get("intent", "").replace("_", " ")
        domain = pair.get("domain", "")

        combined = " ".join(
            [
                question,
                aliases,
                related,
                keywords,
                keywords,
                keywords,
                tags,
                tags,
                intent,
                domain,
                answer,
                steps,
            ]
        )
        return self.preprocessor(combined)

    def _prepare_char_doc(self, pair: dict) -> str:
        parts = [
            pair.get("question", ""),
            " ".join(pair.get("aliases", [])),
            " ".join(pair.get("related_questions", [])),
            " ".join(pair.get("keywords", [])),
            " ".join(pair.get("tags", [])),
            pair.get("answer", ""),
            " ".join(pair.get("answer_steps", [])),
        ]
        return " ".join(part.lower() for part in parts if part)

    def _prepare_term_set(self, pair: dict) -> set[str]:
        terms = set()
        for field in ("question", "answer"):
            terms.update(self.preprocessor.tokenize(pair.get(field, "")))

        for field in ("keywords", "tags", "aliases", "related_questions", "answer_steps"):
            value = pair.get(field, [])
            if isinstance(value, str):
                terms.update(self.preprocessor.tokenize(value))
            else:
                for item in value:
                    terms.update(self.preprocessor.tokenize(item))

        terms.update(self.preprocessor.tokenize(pair.get("intent", "").replace("_", " ")))
        terms.update(self.preprocessor.tokenize(pair.get("domain", "")))
        return terms

    def _build_index(self):
        self.word_docs = [self._prepare_word_doc(pair) for pair in self.qa_pairs]
        self.char_docs = [self._prepare_char_doc(pair) for pair in self.qa_pairs]
        self.term_sets = [self._prepare_term_set(pair) for pair in self.qa_pairs]
        self.phrase_texts = [
            " ".join(
                [
                    pair.get("question", "").lower(),
                    " ".join(pair.get("aliases", [])).lower(),
                    " ".join(pair.get("related_questions", [])).lower(),
                    pair.get("search_text", "").lower(),
                ]
            )
            for pair in self.qa_pairs
        ]
        self.exact_forms = [
            {
                pair.get("question", "").strip().lower(),
                *[alias.strip().lower() for alias in pair.get("aliases", []) if alias.strip()],
            }
            for pair in self.qa_pairs
        ]

        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(self.word_docs)

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        self.char_matrix = self.char_vectorizer.fit_transform(self.char_docs)

        log.info(
            "Hybrid index built: %d documents | word_features=%d | char_features=%d",
            len(self.word_docs),
            len(self.word_vectorizer.get_feature_names_out()),
            len(self.char_vectorizer.get_feature_names_out()),
        )

    def _score_query(self, user_input: str) -> tuple[np.ndarray, str]:
        processed = self.preprocessor(user_input)
        if not processed:
            return np.array([]), processed

        raw_query = user_input.strip().lower()
        query_terms = set(processed.split())

        try:
            word_vec = self.word_vectorizer.transform([processed])
            char_vec = self.char_vectorizer.transform([raw_query])
        except Exception as exc:
            log.error("Vectoriser transform failed: %s", exc)
            return np.array([]), processed

        word_scores = cosine_similarity(word_vec, self.word_matrix).flatten()
        char_scores = cosine_similarity(char_vec, self.char_matrix).flatten()

        lexical_scores = np.array(
            [
                len(query_terms & terms) / max(len(query_terms), 1)
                for terms in self.term_sets
            ],
            dtype=float,
        )

        scores = (word_scores * 0.67) + (char_scores * 0.23) + (lexical_scores * 0.10)

        for idx, pair in enumerate(self.qa_pairs):
            if raw_query in self.exact_forms[idx]:
                scores[idx] = max(scores[idx], 0.995)
                continue

            if len(query_terms) >= 2 and processed and processed in self.phrase_texts[idx]:
                scores[idx] += 0.08

            if pair.get("priority", 1) > 1 and lexical_scores[idx] >= 0.3:
                scores[idx] += min(pair.get("priority", 1) * 0.015, 0.05)

        return np.clip(scores, 0.0, 1.0), processed

    def get_response(self, user_input: str) -> tuple:
        scores, processed = self._score_query(user_input)
        if not processed or scores.size == 0:
            return None, 0.0, None, None

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        log.debug(
            "Best match idx=%d score=%.3f query='%s'",
            best_idx,
            best_score,
            processed,
        )

        if best_score < self.threshold:
            return None, best_score, None, None

        pair = self.qa_pairs[best_idx]
        return (
            pair.get("answer"),
            best_score,
            pair.get("question"),
            pair.get("intent"),
        )

    def top_k(self, user_input: str, k: int = 3) -> list[dict]:
        scores, processed = self._score_query(user_input)
        if not processed or scores.size == 0:
            return []

        top_idxs = np.argsort(scores)[::-1][:k]
        return [
            {
                "question": self.qa_pairs[idx]["question"],
                "intent": self.qa_pairs[idx].get("intent"),
                "score": round(float(scores[idx]), 3),
                "source": self.qa_pairs[idx].get("source", "Unknown"),
                "domain": self.qa_pairs[idx].get("domain", "general"),
            }
            for idx in top_idxs
        ]

    def reload(self, new_qa_pairs: list[dict]) -> None:
        self.qa_pairs = new_qa_pairs
        self._build_index()
        log.info("Hybrid index reloaded with %d QA pairs.", len(new_qa_pairs))
