#!/usr/bin/env python3
"""Supervise one fixed, validated native build using existing local game data."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

# Keep the signed application bundle immutable when importing bundled helpers.
sys.dont_write_bytecode = True


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def alert(message):
    # Direct AppleScript argument, never a shell command. No resource contents
    # or secret values enter the error message.
    escaped = str(message).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    subprocess.run(['/usr/bin/osascript', '-e',
                    f'display alert "FC 27 无法启动" message "{escaped}" buttons {{"好"}}'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--test-run-directory', type=Path)
    parser.add_argument('--live-control', action='store_true')
    parser.add_argument('--audio-callback-stats', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    resources = args.config.resolve().parent
    executable = resources.parent/'MacOS/FC27Runtime'
    if digest(executable) != config['runtime_sha256']:
        raise RuntimeError('应用运行文件与安装记录不一致，请重新部署已验证版本。')
    prepared = Path(config['prepared'])
    for name, expected in config['modules'].items():
        if digest(prepared/f'{name}.bin') != expected:
            raise RuntimeError(f'游戏准备文件 {name} 的版本与安装记录不一致。')
    for path in config['required_paths']:
        if not Path(path).exists():
            raise RuntimeError(f'未找到启动所需的本地路径：{path}')
    if Path(config['source']).stat().st_size != config['source_size_bytes']:
        raise RuntimeError('本地游戏镜像与安装记录不一致。')
    if (prepared/'nxpatch.stamp').read_text().strip() != config['rewrite_stamp']:
        raise RuntimeError('游戏指令改写版本与安装记录不一致。')
    run = args.test_run_directory or Path.home()/'Library/Logs/FC27 Native'/f'{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}'
    run.mkdir(parents=True, exist_ok=False)
    (run/'work').mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith(('NX_', 'TOTK_'))}
    env.update(config['environment'])
    if args.live_control:
        env['NX_TEST_CONTROL'] = str((run/'test-control').resolve())
    if args.audio_callback_stats:
        env['NX_AUDIO_CALLBACK_STATS'] = str((run/'work/audio-observation').resolve())
    os.environ['TOTK_GAME_SLOT_DIR'] = env['TOTK_GAME_SLOT_DIR']
    sys.path.insert(0, str(resources))
    from game_slot import GameSlot
    record = {'app': str(resources.parent.parent), 'validated_run': config['validated_run'],
              'source_tree_sha256': config['source_tree_sha256'], 'profile': config['profile'],
              'runtime_sha256': config['runtime_sha256'], 'modules': config['modules'],
              'started': time.time(), 'run': str(run.resolve())}
    # The upstream wrapper reserves "user" for runtimes that acquire its lock
    # themselves. This runtime inherits the wrapper's lock instead.
    with GameSlot(owner='fc27', purpose='EA SPORTS FC 27') as slot:
        with (run/'host.log').open('wb') as log:
            process = slot.popen([str(executable), *config['arguments']], env=env, cwd=run,
                                 stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            (run/'run.json').write_text(json.dumps(record, indent=2)+'\n')
            subprocess.Popen(['/usr/bin/caffeinate', '-di', '-w', str(process.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.Popen([str(executable.parent/'FC27Launcher'), '--activate-child', str(process.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            record['exit_code'] = process.wait()
            record['seconds'] = time.time()-record['started']
    (run/'run.json').write_text(json.dumps(record, indent=2)+'\n')
    if record['exit_code']:
        alert(f'游戏进程已退出，状态码 {record["exit_code"]}。日志位于：{run}')
    return record['exit_code']


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        try:
            directory = Path.home()/'Library/Logs/FC27 Native'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/'launch-errors.log'
            previous = path.read_text()[-32768:] if path.exists() else ''
            path.write_text(previous+json.dumps({'time': time.time(), 'error': str(error)}, ensure_ascii=False)+'\n')
        except OSError:
            pass
        alert(str(error))
        raise
