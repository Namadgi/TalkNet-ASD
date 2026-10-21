#!/usr/bin/env python3
"""Score clip-level ASD service responses against independent binary labels.

Labels JSON: {"config": {"unit": "clip"}, "labels": [
    {"clip": "example.mp4", "speaking": true}, ...]}
Run: python3 tools/asd_eval/accuracy.py --labels labels.json \
    --run prod=production.json --run candidate=candidate.json --out accuracy.json

Uses the service's existing code, not a newly selected score threshold:
0=speaking; 1=not speaking; 2=no usable face (negative operational decision).
No-face outcomes are counted separately, never silently excluded. Every run
must cover exactly the labeled corpus with valid responses; errors fail closed.
This is NOT the official per-person/per-frame AVA mAP evaluation.
"""

import argparse
import json
import statistics
from pathlib import Path

if __package__:
    from .diff import valid
else:
    from diff import valid


def binary_metrics(tp, tn, fp, fn):
    counts = (tp, tn, fp, fn)
    if any(type(n) is not int or n < 0 for n in counts) or not sum(counts):
        raise ValueError('Expected non-negative confusion counts with non-zero total')

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        'n': sum(counts), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': ratio(tp + tn, sum(counts)),
        'precision': ratio(tp, tp + fp),
        'recall': ratio(tp, tp + fn),
        'f1': ratio(2 * tp, 2 * tp + fp + fn),
        'specificity': ratio(tn, tn + fp),
        'false_positive_rate': ratio(fp, tn + fp),
        'false_negative_rate': ratio(fn, tp + fn),
    }


def index_labels(payload):
    if payload.get('config', {}).get('unit') != 'clip':
        raise ValueError('Labels must explicitly declare config.unit=clip; frame labels need a different adapter')
    result = {}
    for row in payload['labels']:
        clip = row['clip']
        if not isinstance(clip, str) or not clip or clip in result:
            raise ValueError('Empty or duplicate clip ID in labels')
        if type(row.get('speaking')) is not bool:
            raise ValueError(f'{clip}: speaking must be an independently sourced boolean label')
        result[clip] = row['speaking']
    if not result:
        raise ValueError('No labeled clips')
    return result


def score_run(labels, payload):
    rows = payload['results']
    indexed = {r['clip']: r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != set(labels):
        raise ValueError('Run must contain exactly the labeled corpus, without duplicates, missing or extra clips')
    tp = tn = fp = fn = 0
    no_face = {True: [], False: []}
    errors = []
    for clip, truth in labels.items():
        response = indexed[clip]
        if not valid(response):
            raise ValueError(f'{clip}: invalid/error response; do not silently exclude it from accuracy')
        prediction = response['code'] == 0
        if response['code'] == 2:
            no_face[truth].append(clip)
        if truth and prediction:
            tp += 1
        elif truth:
            fn += 1
        elif prediction:
            fp += 1
        else:
            tn += 1
        if truth != prediction:
            errors.append({'clip': clip, 'truth_speaking': truth,
                           'predicted_code': response['code'], 'result': response['result'],
                           'score': response['score']})
    metrics = binary_metrics(tp, tn, fp, fn)
    metrics['no_face'] = {
        'total': sum(len(v) for v in no_face.values()),
        'on_speaking': no_face[True], 'on_not_speaking': no_face[False],
    }
    metrics['misclassified'] = errors
    return metrics


def paired_changes(labels, baseline, candidate):
    before = {r['clip']: r for r in baseline['results']}
    after = {r['clip']: r for r in candidate['results']}
    improved, worsened, code_changes = [], [], []
    for clip, truth in labels.items():
        old_correct = (before[clip]['code'] == 0) == truth
        new_correct = (after[clip]['code'] == 0) == truth
        if not old_correct and new_correct:
            improved.append(clip)
        if old_correct and not new_correct:
            worsened.append(clip)
        if before[clip]['code'] != after[clip]['code']:
            code_changes.append({'clip': clip, 'before': before[clip]['code'], 'after': after[clip]['code']})
    return {'improved': improved, 'worsened': worsened, 'code_changes': code_changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--run', action='append', required=True, metavar='NAME=PATH',
                        help='First run is the reference for paired improvements/regressions')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    labels_payload = json.loads(Path(args.labels).read_text())
    labels = index_labels(labels_payload)
    report = {'label_config': labels_payload['config'], 'positive_class': 'speaking',
              'prediction_rule': 'API code == 0; code 2 remains negative and is separately counted',
              'runs': {}}
    reference = None
    for spec in args.run:
        name, path = spec.split('=', 1)
        if not name or name in report['runs']:
            raise ValueError('Run names must be unique and nonempty')
        run = json.loads(Path(path).read_text())
        row = {'provenance': run.get('config', {}), 'metrics': score_run(labels, run)}
        row['groups'] = {}
        groups = sorted({r.get('group', 'main') for r in labels_payload['labels']})
        for group in groups:
            subset = {r['clip']: r['speaking'] for r in labels_payload['labels']
                      if r.get('group', 'main') == group}
            selected = {'results': [r for r in run['results'] if r['clip'] in subset]}
            row['groups'][group] = score_run(subset, selected)
        known_faces = {r['clip']: r['face_present'] == 'yes'
                       for r in labels_payload['labels'] if r.get('face_present') in ('yes', 'no')}
        row['face_presence_check'] = {
            'n_known': len(known_faces),
            'face_present_but_api_no_face': [r['clip'] for r in run['results']
                if known_faces.get(r['clip']) is True and r['code'] == 2],
            'no_face_but_api_found_track': [r['clip'] for r in run['results']
                if known_faces.get(r['clip']) is False and r['code'] != 2],
            'note': 'Clip-level human visibility versus API usable-track outcome; not frame-level detector accuracy',
        }
        latencies = [r['latency_s'] for r in run['results'] if r.get('latency_s') is not None]
        if latencies:
            row['latency_s'] = {'mean': statistics.mean(latencies),
                                'median': statistics.median(latencies),
                                'max': max(latencies), 'n': len(latencies),
                                'note': 'Client HTTP latency; may include cold starts'}
        if reference is not None:
            row['paired_vs_first_run'] = paired_changes(labels, reference, run)
        else:
            reference = run
        report['runs'][name] = row
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
