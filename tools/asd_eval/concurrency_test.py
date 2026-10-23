#!/usr/bin/env python3
"""Compare identical warm serial/concurrent workloads on one candidate revision.

Client speedup alone does not prove server-stage overlap. Every request carries
a sampled trace ID for subsequent server trace inspection. No tokens are saved.
"""
import argparse
import concurrent.futures
import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import threading
import time
import uuid

import requests


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def token(kind):
    return subprocess.run(
        ['gcloud', 'auth', 'print-' + kind + '-token'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--source', required=True, help='Prior eval used ONLY to select diverse clips')
    parser.add_argument('--out', required=True)
    parser.add_argument('--clips', type=int, default=8)
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text())
    candidates = sorted(
        [r for r in source['results'] if not r.get('error') and r.get('score') is not None],
        key=lambda r: (r['code'], r['score'], r['clip']),
    )
    if not 2 <= args.clips <= len(candidates):
        parser.error('--clips must be between 2 and the number of valid source clips')
    selected = [candidates[round(i * (len(candidates) - 1) / (args.clips - 1))]
                for i in range(args.clips)]
    bucket = source['config']['bucket']
    access = token('access')
    identity = token('identity')
    local = threading.local()
    report = {'config': dict(vars(args), bucket=bucket, started_at=utc_now()), 'phases': []}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')

    def call(clip, slots):
        with slots:
            if not hasattr(local, 'session'):
                local.session = requests.Session()
            trace_id = uuid.uuid4().hex
            span_id = uuid.uuid4().hex[:16]
            record = {'clip': clip['clip'], 'object_name': clip['object_name'],
                      'trace_id': trace_id, 'started_at': utc_now(), 'error': None}
            start = time.perf_counter()
            try:
                response = local.session.post(
                    args.endpoint.rstrip('/') + '/predictions/mono',
                    headers={'Authorization': 'Bearer ' + identity,
                             'traceparent': f'00-{trace_id}-{span_id}-01',
                             'X-Cloud-Trace-Context': f'{trace_id}/{int(span_id, 16)};o=1'},
                    json={'instances': [{'token': access, 'bucket_name': bucket,
                                         'object_name': clip['object_name']}]},
                    timeout=180,
                )
                record['http_status'] = response.status_code
                response.raise_for_status()
                body = response.json()
                if type(body.get('code')) is not int or body['code'] not in (0, 1, 2):
                    raise ValueError('Invalid decision code')
                for field in ('score', 'result'):
                    value = body.get(field)
                    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
                        raise ValueError('Invalid ' + field)
                record['response'] = {k: body[k] for k in ('code', 'score', 'result')}
            except (requests.RequestException, ValueError) as error:
                # Do not persist request headers/body or access tokens.
                record['error'] = type(error).__name__
            record['latency_s'] = time.perf_counter() - start
            record['finished_at'] = utc_now()
            return record

    expected = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        report['warmup'] = pool.submit(call, selected[0], threading.Semaphore(1)).result()
        save()
        if report['warmup']['error']:
            raise SystemExit('Warmup failed; see report. No load test sent.')
        for name, concurrency in [('serial_before', 1), ('parallel_2', 2),
                                  ('parallel_4_a', 4), ('parallel_4_b', 4),
                                  ('serial_after', 1)]:
            slots = threading.Semaphore(concurrency)
            start = time.perf_counter()
            rows = list(pool.map(lambda clip: call(clip, slots), selected))
            wall = time.perf_counter() - start
            mismatches = []
            if name == 'serial_before':
                expected = {r['clip']: r.get('response') for r in rows}
            for row in rows:
                if row['error'] or row.get('response') != expected[row['clip']]:
                    mismatches.append(row['clip'])
            latencies = [r['latency_s'] for r in rows]
            phase = {'name': name, 'concurrency': concurrency, 'wall_s': wall,
                     'requests_per_second': len(rows) / wall,
                     'p50_s': statistics.median(latencies), 'p95_s': percentile(latencies, .95),
                     'errors': sum(r['error'] is not None for r in rows),
                     'mismatches': mismatches, 'results': rows}
            report['phases'].append(phase)
            save()
            print(f'{name}: {wall:.2f}s, {len(rows) / wall:.3f} req/s, '
                  f'p50={phase["p50_s"]:.2f}s p95={phase["p95_s"]:.2f}s, '
                  f'errors={phase["errors"]}, mismatches={len(mismatches)}', flush=True)
            if phase['errors']:
                raise SystemExit('Request errors; stopping load test early.')
    serial_wall = statistics.mean(p['wall_s'] for p in report['phases'] if p['concurrency'] == 1)
    for phase in report['phases']:
        phase['speedup_vs_mean_serial'] = serial_wall / phase['wall_s']
    report['config']['finished_at'] = utc_now()
    report['responses_match_serial'] = all(not p['mismatches'] for p in report['phases'])
    report['server_overlap_verified'] = False  # Requires separate trace/log inspection.
    save()
    if not report['responses_match_serial']:
        raise SystemExit('Responses differed from same-image serial reference; inspect report.')


if __name__ == '__main__':
    main()
