#!/usr/bin/env python3
"""Build source tools only. No game files, keys or firmware are read."""
from pathlib import Path
import argparse
import os
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--debug', action='store_true')
p.add_argument('--test', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
env = dict(os.environ)
env.pop('NX_MODULES_DIR', None)
args = ['cargo', 'test' if a.test else 'build', '--workspace', '--locked']
if not a.debug:
    args.append('--release')
subprocess.run(args, cwd=root, env=env, check=True)
