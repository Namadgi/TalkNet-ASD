"""Ground-truth classification metrics, separate from response parity."""

import unittest

from tools.asd_eval.accuracy import binary_metrics, index_labels, paired_changes, score_run


def response(clip, code, error=None):
    return {'clip': clip, 'code': code, 'result': 50.0 if code == 0 else 0.0,
            'score': 72.22 if code == 0 else 0.0, 'error': error}


class AccuracyTests(unittest.TestCase):
    def test_historical_confluence_counts(self):
        metrics = binary_metrics(tp=52, tn=35, fp=3, fn=9)
        self.assertAlmostEqual(metrics['accuracy'], 87 / 99)
        self.assertAlmostEqual(metrics['precision'], 52 / 55)
        self.assertAlmostEqual(metrics['recall'], 52 / 61)
        self.assertAlmostEqual(metrics['f1'], 104 / 116)

    def test_no_face_is_not_silently_excluded(self):
        metrics = score_run({'positive': True, 'negative': False},
                            {'results': [response('positive', 2), response('negative', 2)]})
        self.assertEqual((metrics['fn'], metrics['tn']), (1, 1))
        self.assertEqual(metrics['no_face']['total'], 2)
        self.assertEqual(metrics['accuracy'], 0.5)

    def test_decision_uses_code_not_remapped_score(self):
        row = dict(response('positive', 0), result=4.26, score=21.3)
        self.assertEqual(score_run({'positive': True}, {'results': [row]})['tp'], 1)

    def test_invalid_responses_and_corpora_fail(self):
        good = response('a', 0)
        for rows in ([], [good, good], [response('b', 0)], [response('a', 0, 'failed')],
                     [dict(good, score=None)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                score_run({'a': True}, {'results': rows})

    def test_undefined_precision_is_not_invented(self):
        self.assertIsNone(binary_metrics(0, 2, 0, 1)['precision'])
        self.assertEqual(binary_metrics(0, 2, 0, 1)['recall'], 0)

    def test_labels_must_be_clip_level_explicit_booleans(self):
        valid = {'config': {'unit': 'clip'}, 'labels': [{'clip': 'a', 'speaking': True}]}
        self.assertEqual(index_labels(valid), {'a': True})
        for payload in ({'config': {'unit': 'frame'}, 'labels': valid['labels']},
                        {'config': {'unit': 'clip'}, 'labels': [{'clip': 'a', 'speaking': 'true'}]},
                        {'config': {'unit': 'clip'}, 'labels': valid['labels'] * 2}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                index_labels(payload)

    def test_paired_improvements_and_no_face_code_changes(self):
        labels = {'a': True, 'b': False, 'c': False}
        before = {'results': [response('a', 1), response('b', 1), response('c', 2)]}
        after = {'results': [response('a', 0), response('b', 0), response('c', 1)]}
        changes = paired_changes(labels, before, after)
        self.assertEqual(changes['improved'], ['a'])
        self.assertEqual(changes['worsened'], ['b'])
        self.assertEqual(len(changes['code_changes']), 3)


if __name__ == '__main__':
    unittest.main()
