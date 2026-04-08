import tempfile
import unittest
from pathlib import Path

import analytics
import app as helpdesk_app
from knowledge_base import (
    KB_META,
    QA_PAIRS,
    find_by_intent,
    kb_summary,
    search_pairs,
)
from nlp_engine import TFIDFChatbot


class HelpdeskTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpdesk_app.app.config.update(
            TESTING=True,
            RATELIMIT_ENABLED=False,
        )

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        analytics.DB_PATH = Path(self.tmpdir.name) / "analytics-test.db"
        analytics.init_db()
        helpdesk_app.cache.clear()
        self.client = helpdesk_app.app.test_client()
        with self.client.session_transaction() as session:
            session.clear()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_kb_meta_matches_actual_pairs(self):
        self.assertEqual(KB_META["total_pairs"], len(QA_PAIRS))
        self.assertEqual(kb_summary()["total_pairs"], len(QA_PAIRS))
        self.assertGreaterEqual(len(kb_summary()["featured_items"]), 1)

    def test_search_pairs_finds_new_payment_issue_entry(self):
        hits = search_pairs("payment not reflected", limit=3)
        intents = [item["intent"] for item in hits]
        self.assertIn("fee_payment_not_reflected", intents)

    def test_hybrid_matcher_handles_typos(self):
        chatbot = TFIDFChatbot(QA_PAIRS, threshold=0.2)
        answer, score, _, intent = chatbot.get_response("hostel maintanance complant")
        self.assertIsNotNone(answer)
        self.assertEqual(intent, "hostel_maintenance_complaint")
        self.assertGreaterEqual(score, 0.2)

    def test_chat_response_includes_steps_and_rich_history(self):
        response = self.client.post(
            "/chat",
            json={"message": "My fee payment is not reflected. What should I do?"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["source"], "tfidf")
        self.assertEqual(payload["intent"], "fee_payment_not_reflected")
        self.assertTrue(payload["steps"])
        self.assertEqual(payload["office"], "Finance Office")

        history = self.client.get("/history").get_json()["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["meta"]["intent"], "fee_payment_not_reflected")
        self.assertEqual(history[1]["meta"]["office"], "Finance Office")

    def test_stats_and_debug_endpoints_return_dynamic_data(self):
        stats = self.client.get("/stats").get_json()
        self.assertEqual(stats["total_qa_pairs"], len(QA_PAIRS))
        self.assertIn("featured_items", stats)
        self.assertTrue(stats["featured_items"])
        self.assertIn("stale_count", stats)

        debug = self.client.get("/admin/debug?q=placement documents").get_json()
        self.assertEqual(debug["knowledge_base"]["summary"]["total_pairs"], len(QA_PAIRS))
        self.assertIn("analytics", debug)
        self.assertIn("search_results", debug)
        self.assertTrue(
            any(item["intent"] == "placement_documents" for item in debug["search_results"])
        )

    def test_find_by_intent_returns_new_document_entry(self):
        pair = find_by_intent("placement_documents")
        self.assertIsNotNone(pair)
        self.assertEqual(pair["source"], "CDC")


if __name__ == "__main__":
    unittest.main()
