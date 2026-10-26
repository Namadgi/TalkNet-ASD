"""Human input normalization must not relabel clips or use model predictions."""
import unittest

from tools.asd_eval.prepare_labels import prepare


class PrepareLabelsTests(unittest.TestCase):
    inventory = {'config': {'bucket': 'test', 'prefix': 'clips/'},
                 'results': [{'clip': 'a.mp4'}, {'clip': 'b_df_1.mp4'}, {'clip': 'c.mp4'}]}

    def source(self, rows):
        return {'config': {'review_window': 'First 10 seconds'}, 'labels': rows}

    def test_normalizes_without_dropping_face_uncertain_or_marking_deepfake_negative(self):
        rows = [{'clip': 'a.mp4', 'speaking': 0, 'face_present': 'uncertain'},
                {'clip': 'b_df_1.mp4', 'speaking': 1, 'face_present': 'yes'},
                {'clip': 'c.mp4', 'speaking': None, 'face_present': None}]
        labels, manifest = prepare(self.source(rows), self.inventory, 'hash')
        self.assertIs(labels['labels'][0]['speaking'], False)
        self.assertIs(labels['labels'][1]['speaking'], True)
        self.assertEqual(labels['labels'][1]['group'], 'deepfake_robustness')
        self.assertEqual(labels['config']['pending'], ['c.mp4'])
        self.assertEqual(manifest['results'][0]['object_name'], 'clips/a.mp4')
        self.assertEqual(rows[0]['speaking'], 0)

    def test_rejects_invalid_labels_and_contradictions(self):
        for value, face in [('1', 'yes'), (2, 'yes'), (1, 'no'), (0, None)]:
            with self.subTest(value=value, face=face), self.assertRaises(ValueError):
                prepare(self.source([{'clip': 'a.mp4', 'speaking': value,
                                      'face_present': face}]), self.inventory, 'hash')

    def test_rejects_duplicates_unknown_and_empty_selection(self):
        valid = {'clip': 'a.mp4', 'speaking': True, 'face_present': 'yes'}
        for rows in ([valid, valid], [dict(valid, clip='unknown')],
                     [dict(valid, speaking=None)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                prepare(self.source(rows), self.inventory, 'hash')


if __name__ == '__main__':
    unittest.main()
