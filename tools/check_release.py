#!/usr/bin/env python3
"""Check the exact source export; this is not legal or copyright clearance."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'SOURCE-MANIFEST.json'
SKIP = {'.git', 'target', '__pycache__'}
ROOT_FILES = {'.gitignore', 'Cargo.toml', 'Cargo.lock', 'LICENSE', 'README.md',
              'THIRD_PARTY_NOTICES.md', 'CONTRIBUTING.md', 'PUBLICATION.md',
              'PROVENANCE.json', 'SOURCE-MANIFEST.json'}
SUFFIXES = {'.rs', '.toml', '.lock', '.html', '.py', '.swift', '.md', '.txt', '.json'}
FORBIDDEN = {'work', 'cache', 'caches', 'runs', 'profiles', 'outputs', 'private',
             'local', 'extracted', 'normalized', 'prepared', 'rewritten', 'firmware', 'appicon.icon'}


def inventory():
    records = []
    errors = []
    for directory, folders, files in os.walk(ROOT):
        folders[:] = sorted(name for name in folders if name not in SKIP)
        for name in sorted(files):
            path = Path(directory) / name
            rel = path.relative_to(ROOT)
            if path == MANIFEST:
                continue
            if path.is_symlink() or any(part.lower() in FORBIDDEN for part in rel.parts):
                errors.append(f'Excluded path: {rel}')
                continue
            if len(rel.parts) == 1:
                allowed = name in ROOT_FILES
            else:
                allowed = rel.parts[0] in {'crates', 'tools', 'launcher', 'docs', 'licenses', 'games'} and path.suffix in SUFFIXES
            if not allowed:
                errors.append(f'Unapproved file type/location: {rel}')
                continue
            data = path.read_bytes()
            if len(data) > 2 * 1024 * 1024:
                errors.append(f'Unexpected source file size: {rel}')
            try:
                source = data.decode('utf-8')
            except UnicodeDecodeError:
                errors.append(f'Non-text payload: {rel}')
                continue
            if '\x00' in source:
                errors.append(f'NUL bytes in source: {rel}')
            if ('/' + 'Users/') in source or re.search(r'[A-Za-z]:\\Users\\', source):
                errors.append(f'Personal absolute path: {rel}')
            # Rules are built in parts so this checker does not match itself.
            secret_names = r'(?:header' + r'_key|master' + r'_key_[0-9a-f]+|key_area' + r'_key_[a-z0-9_]+|title' + r'kek_[0-9a-f]+)'
            if re.search(r'(?im)' + secret_names + r'[\x22\x27]?\s*[:=]\s*[\x22\x27]?[0-9a-f]{32,64}\b', source):
                errors.append(f'Possible literal key assignment: {rel}')
            if re.search(r'[\x22\x27][A-Za-z0-9+/=]{512,}[\x22\x27]', source):
                errors.append(f'Long encoded payload needs review: {rel}')
            records.append({'path': rel.as_posix(), 'bytes': len(data),
                            'sha256': hashlib.sha256(data).hexdigest()})
    return sorted(records, key=lambda row: row['path']), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true',
                        help='Use only after manually reviewing every added/changed file')
    args = parser.parse_args()
    records, errors = inventory()
    if errors:
        raise SystemExit('\n'.join(errors))
    if args.write_manifest:
        MANIFEST.write_text(json.dumps({'scope': 'source-only export; no legal clearance claim',
                                        'files': records}, ensure_ascii=False, indent=2) + '\n')
    else:
        expected = json.loads(MANIFEST.read_text())['files']
        if records != expected:
            raise SystemExit('Source tree differs from the reviewed manifest; review before updating it.')
    print(json.dumps({'files': len(records), 'bytes': sum(row['bytes'] for row in records),
                      'manifest_match': True, 'legal_clearance': False}))


if __name__ == '__main__':
    main()
