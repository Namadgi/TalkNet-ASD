#!/usr/bin/env python3
"""Freeze completed human labels without modifying the original labeling file."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, inventory, source_sha256):
    available = {r['clip']: r for r in inventory['results']}
    seen, labeled, pending = set(), [], []
    for row in source['labels']:
        clip = row['clip']
        if clip in seen or clip not in available:
            raise ValueError(f'Duplicate or unknown clip: {clip}')
        seen.add(clip)
        value, face = row.get('speaking'), row.get('face_present')
        if value is None:
            pending.append(clip)
            continue
        if not (type(value) is bool or (type(value) is int and value in (0, 1))):
            raise ValueError(f'{clip}: speaking must be boolean, 0, 1, or null')
        if face not in ('yes', 'no', 'uncertain'):
            raise ValueError(f'{clip}: completed speaking label needs face_present')
        if bool(value) and face == 'no':
            raise ValueError(f'{clip}: speaking=true contradicts face_present=no')
        labeled.append({'clip': clip, 'speaking': bool(value), 'face_present': face,
                        'group': 'deepfake_robustness' if '_df_' in clip else 'main'})
    if not labeled:
        raise ValueError('No completed labels')
    config = {'unit': 'clip', 'bucket': inventory['config']['bucket'],
              'prefix': inventory['config']['prefix'], 'source_sha256': source_sha256,
              'review_window': source['config']['review_window'],
              'selection': 'Every completed human binary label; face-uncertain entries retained',
              'group_policy': 'Known _df_ filenames reported separately, never treated as negative by default',
              'n_labeled': len(labeled), 'pending': pending,
              'removed_from_inventory': sorted(set(available) - seen)}
    labels = {'config': config, 'labels': labeled}
    manifest = {'config': config, 'results': [
        {'clip': r['clip'], 'object_name': available[r['clip']].get(
            'object_name', config['prefix'] + r['clip'])} for r in labeled]}
    return labels, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--inventory', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    raw = Path(args.labels).read_bytes()
    labels, manifest = prepare(json.loads(raw), json.loads(Path(args.inventory).read_text()),
                               hashlib.sha256(raw).hexdigest())
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('labels.json', labels), ('manifest.json', manifest)]:
        (output / name).write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'labeled': len(labels['labels']),
                      'main': sum(r['group'] == 'main' for r in labels['labels']),
                      'speaking': sum(r['speaking'] for r in labels['labels']),
                      'pending': len(labels['config']['pending']),
                      'removed': labels['config']['removed_from_inventory']}, indent=2))


if __name__ == '__main__':
    main()
