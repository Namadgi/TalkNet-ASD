"""Regression checks for malformed runs that must never pass the golden gate."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.asd_eval import diff


class GoldenGateTests(unittest.TestCase):
    def compare(self, baseline, candidate):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("baseline.json", "candidate.json")]
            for path, rows in zip(paths, (baseline, candidate)):
                path.write_text(json.dumps({"results": rows}))
            with patch("sys.argv", ["diff.py", *map(str, paths)]), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_status:
                    diff.main()
            return exit_status.exception.code

    def setUp(self):
        self.row = dict(clip="test.mp4", code=0, result=90.0, score=90.0,
                        error=None, latency_s=1.0)

    def test_numeric_tolerance_is_inclusive(self):
        candidate = dict(self.row, result=92.0, score=88.0)
        self.assertEqual(self.compare([self.row], [candidate]), 0)
        candidate["score"] = 87.99
        self.assertEqual(self.compare([self.row], [candidate]), 2)

    def test_code_flip_fails(self):
        self.assertEqual(self.compare([self.row], [dict(self.row, code=1)]), 1)

    def test_decimal_rounding_does_not_fail_exact_boundary(self):
        baseline = dict(self.row, result=62.01)
        candidate = dict(self.row, result=64.01)
        self.assertEqual(self.compare([baseline], [candidate]), 0)

    def test_missing_scores_and_nonfinite_scores_fail(self):
        for value in (None, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertEqual(self.compare([self.row], [dict(self.row, score=value)]), 1)

    def test_errors_fail_even_if_both_runs_error(self):
        invalid = dict(self.row, error="HTTP 500")
        self.assertEqual(self.compare([invalid], [invalid]), 1)

    def test_different_corpus_fails(self):
        self.assertEqual(self.compare([self.row], [dict(self.row, clip="other.mp4")]), 1)

    def test_duplicate_clips_are_rejected(self):
        with self.assertRaises(ValueError):
            self.compare([self.row], [self.row, copy.copy(self.row)])


if __name__ == "__main__":
    unittest.main()
