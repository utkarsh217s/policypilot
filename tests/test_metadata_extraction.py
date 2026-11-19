import json
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from app.ingestion.document_loader import load_policy_document
from app.ingestion.metadata_extractor import (
    MetadataError,
    extract_metadata,
    find_labelled_fields,
    parse_date,
)
from app.ingestion.policy_loader import PolicyLoadError

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = ROOT / "data" / "policies" / "remote_work_policy_v1.pdf"

BODY = "1 Scope\nApplies to permanent employees.\n2 Remote Days\nUp to 2 days per week. See Section 1.\n"


class ParsingTests(unittest.TestCase):
    def test_date_formats(self):
        for text in [
            "2026-04-01",
            "1 April 2026",
            "1st Apr 2026",
            "April 1, 2026",
            "01-Apr-2026",
            "13/04/2026",
        ]:
            with self.subTest(text=text):
                parsed, _ = parse_date(text)
                self.assertEqual(
                    parsed,
                    date(2026, 4, 1) if text != "13/04/2026" else date(2026, 4, 13),
                )

    def test_ambiguous_numeric_date_is_day_first_with_warning(self):
        parsed, warning = parse_date("01/04/2026")
        self.assertEqual(parsed, date(2026, 4, 1))
        self.assertIn("ambiguous", warning)

    def test_unreadable_date_stays_unknown(self):
        parsed, warning = parse_date("upon approval")
        self.assertIsNone(parsed)
        self.assertIn("Could not read", warning)

    def test_several_fields_on_one_line(self):
        found = find_labelled_fields(
            ["Policy ID: HR-7 | Version: v2.1 | Effective Date: 2026-01-01"]
        )
        self.assertEqual(
            found,
            {"policy_id": "HR-7", "version": "v2.1", "effective_from": "2026-01-01"},
        )


class ExtractMetadataTests(unittest.TestCase):
    def test_full_header(self):
        result = extract_metadata(
            [
                "Leave Policy",
                "Policy ID: HR-LEAVE",
                "Version: v2.0",
                "Effective Date: 2026-01-01",
                "Valid Until: 2027-01-01",
                "Applies To: Permanent and contract employees",
                "Departments: Engineering; Finance",
                "Jurisdiction: India",
            ],
            source_file="leave.pdf",
        )
        m = result.metadata
        self.assertEqual(
            (m.policy_id, m.title, m.version), ("HR-LEAVE", "Leave Policy", "2.0")
        )
        self.assertEqual(
            (m.effective_from, m.effective_to), (date(2026, 1, 1), date(2027, 1, 1))
        )
        self.assertEqual(
            m.applies_to, ["Permanent and contract employees"]
        )  # not split on "and"
        self.assertEqual(m.departments, ["Engineering", "Finance"])
        self.assertEqual(result.sources["title"], "first line of document")
        self.assertFalse(m.is_synthetic)

    def test_missing_version_is_rejected(self):
        with self.assertRaises(MetadataError) as ctx:
            extract_metadata(
                ["Leave Policy", "Policy ID: HR-LEAVE"], source_file="x.pdf"
            )
        self.assertIn("version", str(ctx.exception))

    def test_policy_id_derived_from_title_with_warning(self):
        result = extract_metadata(
            ["Travel & Expense Policy", "Version: 1.0"], source_file="x.pdf"
        )
        self.assertEqual(result.metadata.policy_id, "TRAVEL-EXPENSE-POLICY")
        self.assertTrue(any("derived from the title" in w for w in result.warnings))

    def test_overrides_win(self):
        result = extract_metadata(
            ["Leave Policy", "Policy ID: HR-LEAVE", "Version: 1.0"],
            source_file="x.pdf",
            overrides={"version": "1.1"},
        )
        self.assertEqual(result.metadata.version, "1.1")
        self.assertEqual(result.sources["version"], "metadata file")

    def test_placeholder_pdf_title_is_ignored(self):
        result = extract_metadata(
            ["Leave Policy", "Version: 1.0"],
            source_file="x.pdf",
            pdf_properties={"title": "Microsoft Word - leave_final.docx"},
        )
        self.assertEqual(result.metadata.title, "Leave Policy")

    def test_end_before_start_is_rejected(self):
        with self.assertRaises(MetadataError):
            extract_metadata(
                [
                    "Leave Policy",
                    "Version: 1.0",
                    "Effective Date: 2026-06-01",
                    "Valid Until: 2026-01-01",
                ],
                source_file="x.pdf",
            )


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_sample_pdf_needs_no_sidecar(self):
        result = load_policy_document(SAMPLE_PDF)
        m = result.document.metadata
        self.assertEqual(
            (m.policy_id, m.version, m.effective_from),
            ("HR-REMOTE", "1.0", date(2026, 4, 1)),
        )
        self.assertEqual(m.departments, ["Engineering", "Finance"])
        self.assertTrue(m.is_synthetic)
        self.assertEqual(len(result.document.clauses), 6)

    def test_docx_with_table_style_header(self):
        from docx import Document as Docx

        doc = Docx()
        doc.add_heading("Expense Policy", 0)
        doc.add_paragraph(
            "Policy ID: FIN-EXP    Version: 3    Effective Date: 15 March 2026"
        )
        doc.add_paragraph("1 Scope")
        doc.add_paragraph("All staff.")
        path = self.tmp / "expense.docx"
        doc.save(path)
        m = load_policy_document(path).document.metadata
        self.assertEqual(
            (m.title, m.policy_id, m.version, m.effective_from),
            ("Expense Policy", "FIN-EXP", "3", date(2026, 3, 15)),
        )

    def test_header_metadata_is_not_stored_as_clause_text(self):
        path = self.write(
            "leave.txt", "Leave Policy\nPolicy ID: HR-LEAVE\nVersion: 1.0\n" + BODY
        )
        clauses = load_policy_document(path).document.clauses
        self.assertFalse(any("Policy ID" in c.text for c in clauses))

    def test_date_line_is_not_mistaken_for_heading(self):
        path = self.write(
            "leave.txt",
            "Leave Policy\nVersion: 1.0\nEffective Date:\n1 April 2026\n" + BODY,
        )
        result = load_policy_document(path)
        self.assertEqual(
            [c.section_number for c in result.document.clauses], ["1", "2"]
        )
        # The value on the line after "Effective Date:" is still read.
        self.assertEqual(result.document.metadata.effective_from, date(2026, 4, 1))

    def test_missing_version_error_names_the_file(self):
        path = self.write("leave.txt", "Leave Policy\nPolicy ID: HR-LEAVE\n" + BODY)
        with self.assertRaises(PolicyLoadError) as ctx:
            load_policy_document(path)
        self.assertIn("leave.txt", str(ctx.exception))
        self.assertIn("Version:", str(ctx.exception))

    def test_sidecar_is_optional_override(self):
        path = self.write(
            "leave.txt", "Leave Policy\nPolicy ID: HR-LEAVE\nVersion: 1.0\n" + BODY
        )
        (self.tmp / "leave.meta.json").write_text(json.dumps({"jurisdiction": "India"}))
        result = load_policy_document(path)
        self.assertEqual(result.document.metadata.jurisdiction, "India")
        self.assertEqual(result.metadata_sources["jurisdiction"], "metadata file")

    def test_sidecar_unknown_field_rejected(self):
        path = self.write("leave.txt", "Leave Policy\nVersion: 1.0\n" + BODY)
        (self.tmp / "leave.meta.json").write_text(json.dumps({"salary": 1}))
        with self.assertRaises(PolicyLoadError):
            load_policy_document(path)


if __name__ == "__main__":
    unittest.main()
