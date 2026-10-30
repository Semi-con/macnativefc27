#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Request the runtime's bounded read-only menu report. Retain evidence locally."""
import argparse
import json
import os
from pathlib import Path
import re
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9-]+', args.label):
        parser.error('Use a lowercase alphanumeric label with hyphens')
    run = args.run.resolve()
    target = run / f'{args.label}-menu-state.json'
    if target.exists():
        raise FileExistsError('Choose a new evidence label')
    record = json.loads((run / 'run.json').read_text())
    environment = record.get('environment', record.get('diagnostics', {}))
    if 'exit_code' in record or environment.get('NX_TEST_CONTROL') != str(run / 'test-control'):
        raise ValueError('Requires a live isolated run with the exact control directory')
    os.kill(record['pid'], 0)
    sequence = time.time_ns()
    request = run / 'test-control.menu-playback'
    temporary = request.with_suffix('.tmp')
    temporary.write_text(f'{sequence}\n')
    temporary.replace(request)
    report = run / f'test-control.menu-playback-{sequence}.json'
    deadline = time.monotonic() + 30
    while True:
        os.kill(record['pid'], 0)
        try:
            data = json.loads(report.read_text())
            break
        except (OSError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError('No completed report yet; inspect local runtime progress')
            time.sleep(.1)
    index = data.get('named_cinematic_candidates', {}).get('manager', {}).get('index', {})
    master = next((row for row in index.get('menu_entries', []) if row.get('instance_name') == 'f26_mm_master_01'), None)
    summary = {'sequence': sequence, 'report': str(report), 'range_complete': data.get('range_complete'),
               'failed_chunks': data.get('failed_chunks'), 'selected_hash': index.get('selected_instance_hash_1884'),
               'selected_menu_name': index.get('selected_menu_name'), 'master': master,
               'limits': 'Raw original state, not interpreted as playback or cleanup completion.'}
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
