"""Regression checks for transitive grouping, reproducibility, and lock protection."""

import json
import tempfile
import unittest
from pathlib import Path

from split_data import BASE, content_groups, prepare_outputs, read_csv, split_rows, write_or_verify


class SplitTests(unittest.TestCase):
    def test_transitive_sharing_of_either_headline(self):
        rows = [
            {"test_id": "1", "headline_A": "First", "headline_B": "Middle!", "eyecatcher_id": "image-1"},
            {"test_id": "2", "headline_A": "middle", "headline_B": "Next", "eyecatcher_id": "image-2"},
            {"test_id": "3", "headline_A": "NEXT?", "headline_B": "Last", "eyecatcher_id": "image-3"},
            {"test_id": "4", "headline_A": "Separate", "headline_B": "Elsewhere", "eyecatcher_id": "image-4"},
        ]
        groups = content_groups(rows)
        self.assertEqual(sorted(sorted(r["test_id"] for r in g) for g in groups.values()), [["1", "2", "3"], ["4"]])
        self.assertEqual(groups, content_groups(list(reversed(rows))))

    def test_image_links_join_headline_components_transitively(self):
        rows = [
            {"test_id": "1", "headline_A": "First", "headline_B": "Second", "eyecatcher_id": "image-1"},
            {"test_id": "2", "headline_A": "Third", "headline_B": "Fourth", "eyecatcher_id": "image-1"},
            {"test_id": "3", "headline_A": "FOURTH!", "headline_B": "Fifth", "eyecatcher_id": "image-2"},
            {"test_id": "4", "headline_A": "Sixth", "headline_B": "Seventh", "eyecatcher_id": "image1"},
        ]
        groups = content_groups(rows)
        self.assertEqual(sorted(sorted(r["test_id"] for r in g) for g in groups.values()), [["1", "2", "3"], ["4"]])
        self.assertEqual(groups, content_groups(list(reversed(rows))))

    def test_missing_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing image"):
            content_groups([{"test_id": "1", "headline_A": "First", "headline_B": "Second", "eyecatcher_id": " "}])

    def test_real_split_matches_recorded_lock_and_has_no_overlap(self):
        outputs, lock = prepare_outputs(BASE / "inputs/upworthy_pairs.csv", BASE / "inputs/reddit_pairs.csv")
        self.assertEqual(lock, json.loads((BASE / "TEST_LOCK.json").read_text()))
        summary = json.loads(outputs["split_summary.json"])
        self.assertEqual(summary["splits"]["dev"]["pairs"], 386)
        self.assertEqual(summary["splits"]["test"]["pairs"], 1543)
        self.assertEqual(summary["splits"]["dev"]["stratum_counts"], summary["dev_stratum_targets"])
        leakage = json.loads(outputs["leakage_report.json"])
        self.assertEqual(leakage["normalized_headline_overlap_dev_test"], 0)
        self.assertEqual(leakage["image_id_overlap_dev_test"], 0)
        self.assertEqual(leakage["content_group_overlap_dev_test"], 0)

    def test_input_row_order_does_not_change_output_bytes(self):
        from split_data import csv_bytes
        fields, rows = read_csv(BASE / "inputs/upworthy_pairs.csv")
        expected, _ = prepare_outputs(BASE / "inputs/upworthy_pairs.csv", BASE / "inputs/reddit_pairs.csv")
        with tempfile.TemporaryDirectory() as temp:
            reordered = Path(temp) / "reordered.csv"
            reordered.write_bytes(csv_bytes(list(reversed(rows)), fields))
            actual, lock = prepare_outputs(reordered, BASE / "inputs/reddit_pairs.csv")
            self.assertEqual(expected, actual)
            self.assertNotEqual(lock["inputs_sha256"]["upworthy_pairs.csv"], json.loads((BASE / "TEST_LOCK.json").read_text())["inputs_sha256"]["upworthy_pairs.csv"])

    def test_split_preserves_every_source_row_and_value(self):
        _, source = read_csv(BASE / "inputs/upworthy_pairs.csv")
        splits, _, _ = split_rows(source)
        actual = [
            {key: value for key, value in row.items() if key != "content_group_id"}
            for part in splits.values() for row in part
        ]
        self.assertEqual(sorted(actual, key=lambda r: r["test_id"]), sorted(source, key=lambda r: r["test_id"]))
        self.assertFalse({r["test_id"] for r in splits["dev"]} & {r["test_id"] for r in splits["test"]})

    def test_lock_drift_and_file_tampering_fail_before_any_write(self):
        from split_data import json_bytes
        outputs = {"test.csv": b"locked\n", "summary.json": b"{}\n"}
        lock = {"version": 1}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            manifest = folder / "lock.json"
            write_or_verify(outputs, lock, folder, manifest)
            with self.assertRaisesRegex(ValueError, "LOCK MISMATCH"):
                write_or_verify(outputs, {"version": 2}, folder, manifest)
            self.assertEqual(manifest.read_bytes(), json_bytes(lock))
            (folder / "test.csv").write_bytes(b"tampered\n")
            (folder / "summary.json").unlink()
            with self.assertRaisesRegex(ValueError, "LOCKED FILE MISMATCH"):
                write_or_verify(outputs, lock, folder, manifest)
            self.assertFalse((folder / "summary.json").exists())
            self.assertEqual((folder / "test.csv").read_bytes(), b"tampered\n")


if __name__ == "__main__":
    unittest.main()
