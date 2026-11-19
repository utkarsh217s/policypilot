from copy import deepcopy
import json
from pathlib import Path
import unittest

from pydantic import ValidationError

from app.schemas.policy import PolicyClause, PolicyDocument, PolicyMetadata

ROOT = Path(__file__).resolve().parents[1]


class PolicySchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the same synthetic fixture as the manual validation script.
        cls.fixture = json.loads(
            (ROOT / "data/policies/synthetic_leave_v1.json").read_text(
                encoding="utf-8-sig"
            )
        )

    def setUp(self):
        # Each test gets independent data so mutations cannot leak between tests.
        self.data = deepcopy(self.fixture)

    def test_valid_document(self):
        document = PolicyDocument.model_validate(self.data)
        self.assertEqual(document.metadata.policy_id, "HR-LEAVE")
        self.assertTrue(document.metadata.is_synthetic)
        self.assertEqual(len(document.clauses), 4)

    def test_blank_clause_text_rejected(self):
        self.data["clauses"][0]["text"] = "   "
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_invalid_date_range_rejected(self):
        self.data["metadata"]["effective_to"] = "2025-12-31"
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_equal_effective_dates_rejected(self):
        self.data["metadata"]["effective_to"] = self.data["metadata"]["effective_from"]
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_duplicate_sections_rejected(self):
        self.data["clauses"].append(deepcopy(self.data["clauses"][0]))
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_policy_mismatch_rejected(self):
        self.data["clauses"][0]["policy_id"] = "OTHER-POLICY"
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_version_mismatch_rejected(self):
        self.data["clauses"][0]["version"] = "2.0"
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_empty_document_rejected(self):
        self.data["clauses"] = []
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_page_zero_rejected(self):
        self.data["clauses"][0]["page_number"] = 0
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_reversed_page_range_rejected(self):
        self.data["clauses"][0].update(page_number=3, page_end=2)
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_page_end_without_start_rejected(self):
        self.data["clauses"][0].update(page_number=None, page_end=2)
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_unknown_field_rejected(self):
        self.data["metadata"]["unexpected_field"] = True
        with self.assertRaises(ValidationError):
            PolicyDocument.model_validate(self.data)

    def test_clause_id_is_deterministic(self):
        clause_data = self.data["clauses"][0]
        first = PolicyClause.model_validate(clause_data)
        second = PolicyClause.model_validate(deepcopy(clause_data))
        self.assertEqual(first.clause_id, second.clause_id)

    def test_clause_id_changes_with_version(self):
        first = PolicyClause.model_validate(self.data["clauses"][0])
        revised = deepcopy(self.data["clauses"][0])
        revised["version"] = "2.0"
        second = PolicyClause.model_validate(revised)
        self.assertNotEqual(first.clause_id, second.clause_id)

    def test_missing_applicability_remains_unspecified(self):
        metadata = deepcopy(self.data["metadata"])
        metadata.pop("departments", None)
        metadata.pop("applies_to", None)
        parsed = PolicyMetadata.model_validate(metadata)
        self.assertEqual(parsed.departments, [])
        self.assertEqual(parsed.applies_to, [])

    def test_json_export_includes_clause_ids(self):
        document = PolicyDocument.model_validate(self.data)
        exported = json.loads(document.model_dump_json())
        self.assertIn("clause_id", exported["clauses"][0])


if __name__ == "__main__":
    unittest.main()
