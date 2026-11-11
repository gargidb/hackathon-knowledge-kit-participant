import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))

from harness.run_eval import load_snapshots  # noqa: E402
from solution import SearchKnowledge  # noqa: E402


class SearchKnowledgeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = SearchKnowledge()
        for snapshot in load_snapshots(ROOT / "snapshots"):
            cls.knowledge.ingest(snapshot)

    def test_specific_certificate_term_excludes_other_terms(self):
        answer = self.knowledge.answer(
            "What is the APY on the 17-month fixed certificate special?"
        )

        self.assertIn("4.10%", answer.text)
        self.assertIn("4.20%", answer.text)
        self.assertNotIn("10 Month", answer.text)

    def test_broad_certificate_query_returns_multiple_terms(self):
        answer = self.knowledge.answer(
            "Get me the rate for the certificate of accounts."
        )

        self.assertIn("17 Month Fixed Certificate Special", answer.text)
        self.assertIn("10 Month Fixed Jumbo Certificate Special", answer.text)
        self.assertIn("3.90%", answer.text)

    def test_as_of_query_uses_historical_certificate_rate(self):
        answer = self.knowledge.answer(
            "What was the rate on the 10-month fixed jumbo special certificate as of May 2026?",
            as_of="2026-05-31",
        )

        self.assertIn("3.90%", answer.text)
        self.assertNotIn("4.00%", answer.text)

    def test_mortgage_rate_query_includes_requested_tiers(self):
        answer = self.knowledge.answer(
            "Can you tell me the interest rate for a 30-year mortgage?"
        )

        self.assertIn("7.13%", answer.text)
        self.assertIn("7.25%", answer.text)
        self.assertNotIn("6.037%", answer.text)

    def test_auto_rate_cites_the_current_auto_rate_page(self):
        answer = self.knowledge.answer(
            "What's the APR for a 60-month auto loan on a 2025 model-year vehicle?"
        )

        self.assertIn("5.39%", answer.text)
        self.assertIn(
            "https://www.dupaco.com/personal/loans/auto-motorcycle-loans",
            {source.url.rstrip("/") for source in answer.sources},
        )

    def test_galena_branch_address_is_retrieved(self):
        answer = self.knowledge.answer("What is the address of the Galena Branch?")

        self.assertIn("11375 Oldenburg", answer.text)
        self.assertIn("61036", answer.text)

    def test_broad_lobby_hours_include_saturday_schedule(self):
        answer = self.knowledge.answer(
            "Do all Dupaco branches keep the same lobby hours?"
        )

        self.assertIn("12:30", answer.text)
        self.assertIn("noon", answer.text.lower())

    def test_general_drive_up_hours_cite_a_schedule_with_weekend_variation(self):
        answer = self.knowledge.answer("What are the drive-up hours?")

        self.assertIn("8 a.m.", answer.text)
        self.assertIn("6 p.m.", answer.text)
        self.assertIn("9 a.m.", answer.text)
        self.assertIn("noon", answer.text.lower())
        self.assertTrue(
            any(
                source.url.rstrip("/")
                == "https://www.dupaco.com/newsroom/dupaco-celebrates-completion-of-newest-learning-lab"
                for source in answer.sources
            )
        )

    def test_vto_employee_counts_are_returned_with_their_reported_years(self):
        answer = self.knowledge.answer(
            "How many Dupaco employees used Volunteer Time Off last year?"
        )

        self.assertIn("294 Dupaco employees", answer.text)
        self.assertIn("in 2023", answer.text)
        self.assertIn("256 Dupaco employees", answer.text)
        self.assertIn("in 2024", answer.text)
        self.assertEqual(len(answer.sources), 2)


if __name__ == "__main__":
    unittest.main()
