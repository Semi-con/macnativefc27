#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Launch an isolated source build using an existing, verified local preparation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time

from prepare_fc27 import ROOT, digest

sys.path.insert(0, str(ROOT / 'launcher'))
from game_slot import GameSlot, Preempted

BASE_FLAGS = {
    'NX_WINDOWED': '1', 'NX_PHONE_PADS': '0', 'NX_NET': 'off',
    'NX_TRACE_SYMS': 'nnMain,nninitStartup', 'NX_MEMORY_BUDGET_MIB': '3285',
    'NX_FPS': '1', 'NX_NVN_LOG': '1000',
    'NX_DEFER_EVENT_WAITS': '1', 'NX_DEFER_TRANSFER_WAITS': '1', 'NX_DEFER_SHADER_WAITS': '1',
    'NX_ISOLATE_SHADER_FAULTS': '1', 'NX_GPU_TIMESTAMP_REPORTS': '1',
    'NX_PAGED_GMEM': '1', 'NX_GRAPHICS_STRUCTURED': '1', 'NX_BUFFER_FORMATS': '1',
    'NX_FC27_VIDEO_TRACE': '1', 'NX_DIRECT_BUFFER_READ': '1', 'NX_BUFFER_READ_FAST': '1',
    'NX_HOST_WAIT_FASTPATHS': '1', 'NX_DEFER_WINDOW_WAITS': '1', 'NX_GPU_POOL_DOMAINS': '1',
    'NX_FC27_RENDER_FPS': '60', 'NX_ARGUMENT_ENCODER_CACHE': '1',
    'NX_INDEXED_POOL_LOOKUP': '1', 'NX_APPROX_MUFU': '1', 'NX_VECTOR_GMEM': '1', 'NX_EMBEDDED_C1': '1',
}
MENU_FLAGS = {
    'NX_FC27_MENU_SEQUENCE': 'g4', 'NX_FC27_MENU_EVENT_TRACE': '1',
    'NX_FC27_MENU_PRIME_PROBE': '1', 'NX_FC27_MENU_AUTO_LIFECYCLE': '1',
    'NX_FC27_MENU_AMBASSADOR_SOURCE': '1', 'NX_FC27_MENU_TIME_TRACE': '1',
    'NX_FC27_MENU_SHADOW_PARENT_BINDINGS': '1',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True, help='Root produced by prepare_fc27.py')
    parser.add_argument('--firmware', type=Path, required=True)
    parser.add_argument('--dlc', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True, help='New external run directory')
    parser.add_argument('--bin-dir', type=Path, default=ROOT / 'target/release')
    parser.add_argument('--profile-source', type=Path, help='Optional user-owned profile copied into this run')
    parser.add_argument('--menu-experiment', action='store_true')
    parser.add_argument('--seconds', type=int, default=1200)
    args = parser.parse_args()
    if not 30 <= args.seconds <= 3600:
        parser.error('--seconds must be between 30 and 3600')
    if args.menu_experiment and args.profile_source is None:
        parser.error('Complete first initialization without --menu-experiment, quit normally, then use --profile-source with that initialized profile. A DATA file alone does not prove initialization is complete.')
    out, prepared = args.out.resolve(), args.prepared.resolve()
    if out.is_relative_to(ROOT) or out.is_relative_to(prepared):
        parser.error('Use a new run directory outside the checkout and preparation directory')
    if not args.firmware.is_dir():
        parser.error('Firmware directory is missing')
    preparation = json.loads((prepared / 'preparation.json').read_text())
    for name, expected in preparation['modules'].items():
        if digest(prepared / 'prepared' / f'{name}.bin') != expected:
            raise ValueError(f'Prepared module {name} failed checksum')
    if digest(prepared / 'normalized/main.npdm') != preparation['npdm_sha256']:
        raise ValueError('Prepared NPDM failed checksum')
    if args.menu_experiment:
        from prepare_menu import EXPECTED
        if not preparation['menu_experiment']:
            raise ValueError('Preparation did not include the menu experiment')
        for path, expected in EXPECTED.items():
            if digest(prepared / 'menu-overlay/romfs' / path.lstrip('/')) != expected:
                raise ValueError('Prepared menu overlay failed checksum')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'work').mkdir()
    profile = out / 'profile'
    if args.profile_source:
        shutil.copytree(args.profile_source, profile)
    else:
        profile.mkdir()
    app = out / 'FC27 Source Test.app'
    macos = app / 'Contents/MacOS'
    macos.mkdir(parents=True)
    executable = macos / 'FC27-test'
    shutil.copy2(args.bin_dir / 'nxhost', executable)
    with (app / 'Contents/Info.plist').open('wb') as stream:
        plistlib.dump({'CFBundleIdentifier': 'local.macnativefc27.test',
                      'CFBundleExecutable': executable.name, 'CFBundleName': 'FC27 Source Test',
                      'CFBundlePackageType': 'APPL', 'CFBundleVersion': '1',
                      'NSHighResolutionCapable': True, 'NSPrincipalClass': 'NSApplication'}, stream)
    subprocess.run(['codesign', '--force', '--sign', '-', str(app)], check=True, capture_output=True)
    subprocess.run(['codesign', '--verify', '--strict', str(app)], check=True, capture_output=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NX_', 'TOTK_'))}
    env.update(BASE_FLAGS)
    if args.menu_experiment:
        env.update(MENU_FLAGS)
    env.update({'NX_TEST_CONTROL': str(out / 'test-control'),
                'NX_SHADER_CACHE': str(out / 'cache/shaders'),
                'NX_PIPELINE_LIST': str(out / 'cache/pipelines.tsv'),
                'NX_AUDIO_CALLBACK_STATS': str(out / 'work/audio-observation'),
                'TOTK_GAME_SLOT_DIR': str(Path.home() / 'Library/Application Support/MacNativeGames/TOTK/game-slot')})
    os.environ['TOTK_GAME_SLOT_DIR'] = env['TOTK_GAME_SLOT_DIR']
    argv = [str(executable), '--patched', str(prepared / 'prepared'), '--npdm', str(prepared / 'normalized/main.npdm')]
    for flag in ('base', 'update', 'keys'):
        argv += ['--' + flag, preparation[flag]]
    for flag, path in [('save', profile / 'save'), ('sdmc', profile / 'sdmc'), ('system', out / 'system'), ('firmware', args.firmware.resolve())]:
        argv += ['--' + flag, str(path)]
    for path in args.dlc:
        argv += ['--dlc', str(path.resolve())]
    if args.menu_experiment:
        argv += ['--mod', str(prepared / 'menu-overlay')]
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    record = {'runtime_sha256': digest(executable), 'source_commit': commit.stdout.strip(),
              'source_manifest_sha256': digest(ROOT / 'SOURCE-MANIFEST.json'),
              'modules': preparation['modules'], 'environment': {k: v for k, v in env.items() if k.startswith('NX_')},
              'profile': str(profile), 'fresh_shader_cache': True,
              'acceptance': 'Inspect actual frames and interactions; startup and exit are insufficient.'}
    with GameSlot(owner='fc27', purpose='Source reproducibility test', expect_seconds=args.seconds+30) as slot:
        with (out / 'host.log').open('wb') as log:
            process = slot.popen(argv, env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT)
            subprocess.Popen(['caffeinate', '-di', '-w', str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            record['pid'] = process.pid
            (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps({'run': str(out), 'pid': process.pid}), flush=True)
            start, reason = time.monotonic(), 'process_exit'
            while process.poll() is None:
                try:
                    slot.check()
                except Preempted:
                    reason = 'user_preemption'
                    break
                if time.monotonic()-start >= args.seconds or log.tell() >= 64*1024*1024:
                    reason = 'time_or_log_limit'
                    break
                time.sleep(.2)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            record.update(exit_code=process.wait(), reason=reason, seconds=time.monotonic()-start)
    (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    return record['exit_code']


if __name__ == '__main__':
    sys.exit(main())
