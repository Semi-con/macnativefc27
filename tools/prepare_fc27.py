#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prepare a supported local dump outside this source repository. No downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def run(argv):
    subprocess.run([str(x) for x in argv], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--update', type=Path, required=True)
    parser.add_argument('--keys', type=Path, required=True)
    parser.add_argument('--decoder-python', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, default=ROOT / 'target/release')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--menu-experiment', action='store_true')
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.is_relative_to(ROOT):
        parser.error('--out must be outside the source checkout')
    for path in (args.base, args.update, args.keys, args.decoder_python):
        if not path.is_file():
            parser.error('A required local input is missing')
    args.out.mkdir(parents=True, exist_ok=False)
    identity = json.loads((ROOT / 'games/fc27/identity.json').read_text())
    bins = args.bin_dir.resolve()
    extracted, normalized, patched = [args.out / n for n in ('exefs-input', 'normalized', 'prepared')]
    run([bins / 'nxfs', '--keys', args.keys, 'extract', '--base', args.base,
         '--update', args.update, '--out', extracted])
    npdm = extracted / 'exefs/main.npdm'
    if digest(npdm) != identity['npdm_sha256']:
        raise ValueError('NPDM version mismatch; no rewrite performed')
    with (extracted / 'exefs/main').open('rb') as stream:
        stream.seek(0x40)
        if stream.read(32).hex() != identity['main_build_id']:
            raise ValueError('Main build ID mismatch; no rewrite performed')
    # This is a separate standalone GPL-2.0 tool, with an explicit filesystem
    # interface. The Rust GPL-3.0 runtime never links/imports its decoder.
    run([args.decoder_python, ROOT / 'tools/normalize_nso.py', '--source', extracted / 'exefs', '--out', normalized])
    renderer = normalized / 'renderer'
    run([bins / 'nxfs', '--keys', args.keys, 'romfs-file', '--base', args.base,
         '--update', args.update, '--file', '/Engine.Render.Core2.PlatformNvn.nro', '--out', renderer])
    if digest(renderer) != '487a417b97839e2e554a61caf3c4a61e0739078a63dbbd8b85c35a4cf1a1c714':
        raise ValueError('Updated renderer hash mismatch')
    shutil.copyfile(npdm, normalized / 'main.npdm')
    run([bins / 'nxpatch', 'rewrite', *[normalized / n for n in ('main', 'rtld', 'sdk', 'renderer')], '--out', patched])
    modules = {n: digest(patched / f'{n}.bin') for n in ('main', 'rtld', 'sdk', 'renderer')}
    if modules != identity['rewritten_module_sha256_from_private_development_run']:
        raise ValueError('Rewritten module outputs differ from the recorded development inputs')
    if args.menu_experiment:
        run([sys.executable, ROOT / 'tools/prepare_menu.py', '--base', args.base,
             '--update', args.update, '--keys', args.keys, '--bin-dir', bins,
             '--out', args.out / 'menu-overlay'])
    record = {'schema': 1, 'base': str(args.base.resolve()), 'update': str(args.update.resolve()),
              'keys': str(args.keys.resolve()), 'modules': modules,
              'npdm_sha256': digest(npdm), 'rewrite_stamp': (patched / 'nxpatch.stamp').read_text().strip(),
              'menu_experiment': args.menu_experiment,
              'acceptance': 'Preparation hashes only; game execution must be verified separately.'}
    (args.out / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Local preparation complete; no files uploaded.')


if __name__ == '__main__':
    main()
