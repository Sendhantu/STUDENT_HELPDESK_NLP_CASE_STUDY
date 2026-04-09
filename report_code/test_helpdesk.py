import tempfile, unittest
from pathlib import Path
import analytics, app as helpdesk_app
from knowledge_base import KB_META, QA_PAIRS, find_by_intent, kb_summary, search_pairs
from nlp_engine import TFIDFChatbot

class HelpdeskTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helpdesk_app.app.config.update(TESTING=True, RATELIMIT_ENABLED=False)

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        analytics.DB_PATH = Path(self.tmpdir.name) / "analytics-test.db"
        analytics.init_db()
        helpdesk_app.cache.clear()
        self.client = helpdesk_app.app.test_client()

    def tearDown(self):
        self.tmpdir.cleanup()

    # Test 1: KB metadata matches actual pair count
    def test_kb_meta_matches_actual_pairs(self):
        self.assertEqual(KB_META["total_pairs"], len(QA_PAIRS))
        self.assertGreaterEqual(len(kb_summary()["featured_items"]), 1)

    # Test 2: Search finds specific intent by keyword
    def test_search_finds_payment_issue(self):
        hits = search_pairs("payment not reflected", limit=3)
        self.assertIn("fee_payment_not_reflected", [h["intent"] for h in hits])

    # Test 3: TF-IDF handles typos via fuzzy matching
    def test_hybrid_matcher_handles_typos(self):
        chatbot = TFIDFChatbot(QA_PAIRS, threshold=0.2)
        answer, score, _, intent = chatbot.get_response("hostel maintanance complant")
        self.assertEqual(intent, "hostel_maintenance_complaint")
        self.assertGreaterEqual(score, 0.2)

    # Test 4: Chat returns steps, intent, office + correct session history
    def test_chat_response_with_steps_and_history(self):
        res = self.client.post("/chat",
            json={"message": "My fee payment is not reflected. What should I do?"})
        payload = res.get_json()
        self.assertEqual(payload["source"], "tfidf")
        self.assertEqual(payload["intent"], "fee_payment_not_reflected")
        self.assertTrue(payload["steps"])
        self.assertEqual(payload["office"], "Finance Office")

        history = self.client.get("/history").get_json()["history"]
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["meta"]["intent"], "fee_payment_not_reflected")

    # Test 5: /stats and /admin/debug return live dynamic data
    def test_stats_and_debug_endpoints(self):
        stats = self.client.get("/stats").get_json()
        self.assertEqual(stats["total_qa_pairs"], len(QA_PAIRS))
        self.assertIn("stale_count", stats)

        debug = self.client.get("/admin/debug?q=placement documents").get_json()
        self.assertTrue(any(i["intent"] == "placement_documents"
                            for i in debug["search_results"]))

    # Test 6: find_by_intent resolves placement_documents to CDC
    def test_find_by_intent_placement_documents(self):
        pair = find_by_intent("placement_documents")
        self.assertIsNotNone(pair)
        self.assertEqual(pair["source"], "CDC")

if __name__ == "__main__":
    unittest.main()
