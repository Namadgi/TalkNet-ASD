#!/usr/bin/env python3
"""Fetch sampled test traces and measure server-stage overlap, without tokens."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

import requests


def timestamp(value):
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def peak(intervals):
    events = [(start, 1) for start, end in intervals]
    events += [(end, -1) for start, end in intervals]
    current = maximum = 0
    for _, delta in sorted(events):
        current += delta
        maximum = max(maximum, current)
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--out', required=True)
    parser.add_argument('--project', default='biometry-416410')
    args = parser.parse_args()
    run = json.loads(Path(args.run).read_text())
    access = subprocess.run(['gcloud', 'auth', 'print-access-token'], check=True,
                            capture_output=True, text=True).stdout.strip()
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + access
    report = {'run': args.run, 'phases': [], 'missing_traces': []}
    for phase in run['phases']:
        spans = []
        for row in phase['results']:
            trace_id = row['trace_id']
            response = session.get(
                f'https://cloudtrace.googleapis.com/v1/projects/{args.project}/traces/{trace_id}',
                timeout=30,
            )
            if response.status_code == 404:
                report['missing_traces'].append(trace_id)
                continue
            response.raise_for_status()
            for span in response.json().get('spans', []):
                if span['name'] in ('preprocess', 'inference', 'mono_process_video'):
                    spans.append({'trace_id': trace_id, 'name': span['name'],
                                  'start': span['startTime'], 'end': span['endTime'],
                                  'labels': span.get('labels', {})})
        stages = {}
        for name in ('preprocess', 'inference', 'mono_process_video'):
            intervals = [(timestamp(s['start']), timestamp(s['end']))
                         for s in spans if s['name'] == name]
            stages[name] = {'count': len(intervals), 'peak_overlap': peak(intervals)}
        cpu_gpu_overlap = any(
            cpu['trace_id'] != gpu['trace_id']
            and max(timestamp(cpu['start']), timestamp(gpu['start']))
            < min(timestamp(cpu['end']), timestamp(gpu['end']))
            for cpu in spans if cpu['name'] == 'preprocess'
            for gpu in spans if gpu['name'] == 'inference'
        )
        summary = {'name': phase['name'], 'stages': stages,
                   'preprocess_overlaps_other_request_inference': cpu_gpu_overlap, 'spans': spans}
        report['phases'].append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'spans'}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print('Missing traces:', len(report['missing_traces']))


if __name__ == '__main__':
    main()
