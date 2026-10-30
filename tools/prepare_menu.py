#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Rebuild the guarded 232 menu experiment from locally extracted original data.

Only identifiers, offsets and checksums are embedded. Original and modified
assets stay outside the checkout. All four output files must match the tested
development overlay. These checks establish byte identity, not game acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from prepare_fc27 import ROOT, digest, run

DATA = '/Data/NX/superbundlelayout/installpackage_02/cas_01.cas'
PATCH = '/Patch/NX/superbundlelayout/installpackage_02/cas_01.cas'
TOC = '/Data/NX/fc/fcgame/fcgame.toc'
PATCH_TOC = '/Patch/NX/fc/fcgame/fcgame.toc'
# Whole-file guards lock this small recipe to the exact analyzed resource set.
ORIGINALS = {
    DATA: '45860709fc908b966691931bdf94fd00a893e1aca2e40cdebaccc60b98339e17',
    PATCH: '032587f333e8e449f59585dd041092cad54c1753b59e8f2b7694ff3c8924c3a3',
    TOC: '1e6630097afad9afc8ad3e14ae351063bf91e259db6c97cf48cc853fcac78172',
    PATCH_TOC: '81364b8baee6580b6eb504beacc1ed023f7589d5160a8acf712fa71abaa539cd',
}
EXPECTED = {
    DATA: '6586a5b8d596788743a4457229a08a531666051453de78d8379ae2a3e77de9bf',
    PATCH: '142aada755089e9a4e2f1ab31a8e0baa8897e179010d2391aad8fda5488cf4b8',
    TOC: '25352ebb48f82453008a850c79dbe3a5cc2e5941ceda41429fdce70db23e4f99',
    PATCH_TOC: '16e444e860e3d45576f5afb1ff7ce247a7c1cbc351949ce0a998b4f5493e7c0a',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(bins, source, out, size):
    run([bins / 'fc27-local-cas-decode', source, out, size])
    return out.read_bytes()


def modify_frontend(original):
    require(hashlib.sha256(original).hexdigest() == '37a507bcb236db3d72bb2258329d1a268119c7626c99d55c739f2a34a5235c91', 'Frontend asset mismatch')
    result = bytearray(original)
    nx, g4 = 0x3810fe13, 0x3a30fc92
    for offset, old, new in [(1608, nx, g4), (1640, nx, g4), (1672, g4, nx), (1704, g4, nx)]:
        require(struct.unpack_from('<I', original, offset)[0] == old, 'Event route mismatch')
        struct.pack_into('<I', result, offset, new)
    # Reuse the existing automatic sublevel reference. Its final string grows
    # by one aligned block; offsets after EBXD are shifted with their chunks.
    old_name, new_name = b'FC/FCGame/HapticsFE\0', b'FC/fcgame/MainMenuCinematics\0'
    at = original.find(old_name)
    require(at == 3187 and original.find(old_name, at+1) < 0, 'Sublevel name mismatch')
    require(original[12:16] == b'EBXD' and struct.unpack_from('<I', original, 16)[0] == 3187, 'EBXD chunk mismatch')
    replacement = new_name + bytes(len(old_name) + 16 - len(new_name))
    result = result[:at] + replacement + result[at+len(old_name):]
    struct.pack_into('<I', result, 4, len(result)-8)
    struct.pack_into('<I', result, 16, 3187+16)
    require(struct.unpack_from('<ii', original, 3032) == (1, 8), 'Platform array mismatch')
    struct.pack_into('<i', result, 3036, 7)
    require(hashlib.sha256(result).hexdigest() == 'f2fdbe70d8c7c3808c495b6ba434f4157f64903b1d9dc04bdf4d9b2fa82095d5', 'Frontend result mismatch')
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'update', 'keys', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, default=ROOT / 'target/release')
    args = parser.parse_args()
    out, bins = args.out.resolve(), args.bin_dir.resolve()
    require(not out.is_relative_to(ROOT), 'Output must be outside checkout')
    out.mkdir(parents=True, exist_ok=False)
    romfs = out / 'romfs'
    files = {}
    for name, expected in ORIGINALS.items():
        dest = romfs / name.lstrip('/')
        dest.parent.mkdir(parents=True, exist_ok=True)
        run([bins / 'nxfs', '--keys', args.keys, 'romfs-file', '--base', args.base,
             '--update', args.update, '--file', name, '--out', dest])
        require(digest(dest) == expected, f'Original resource mismatch: {name}')
        files[name] = dest
    # Compressed blocks are read exclusively from the user's verified copy.
    assets = []
    for name, offset, size, decoded_size in [('frontend', 385087177, 1839, 5036), ('master', 387408470, 520, 802)]:
        with files[DATA].open('rb') as stream:
            stream.seek(offset)
            block = stream.read(size)
        source = out / f'{name}-input.casblock'
        source.write_bytes(block)
        decoded = decode(bins, source, out / f'{name}-input.ebx', decoded_size)
        assets.append((block, decoded))
    frontend = modify_frontend(assets[0][1])
    original = assets[1][1]
    require(hashlib.sha256(original).hexdigest() == 'a4ea3854029ef7a84d588a7a93162ff3dff18bd18ff6684bf569e0673c0694dc' and original[205] == 0, 'Master asset mismatch')
    master = bytearray(original)
    master[205] = 1  # AutomaticPrepSchematics; other bytes preserved.
    require(hashlib.sha256(master).hexdigest() == 'c56d2cb932a4b01d7df788070365605419adf3f61aef0efa90447e97e5873491', 'Master result mismatch')
    blocks = [struct.pack('>II', len(b), 0x00700000 | len(b)) + b for b in (frontend, master)]
    with files[DATA].open('ab') as stream:
        offsets = []
        for block in blocks:
            offsets.append(stream.tell())
            stream.write(block)
    toc_edits = [(TOC, 1071782, 0, 385087177), (TOC, 1073928, 1, 387408470), (PATCH_TOC, 926504, 1, 387408470)]
    for name, at, index, old_offset in toc_edits:
        with files[name].open('r+b') as stream:
            stream.seek(at)
            require(struct.unpack('>II', stream.read(8)) == (old_offset, len(assets[index][0])), 'TOC reference mismatch')
            stream.seek(at)
            stream.write(struct.pack('>II', offsets[index], len(blocks[index])))
    for name, at, index in [(DATA, 385081434, 0), (DATA, 385089918, 1), (PATCH, 15630978, 1)]:
        with files[name].open('r+b') as stream:
            stream.seek(at)
            require(stream.read(20) == hashlib.sha1(assets[index][0]).digest(), 'Metadata checksum mismatch')
            stream.seek(at)
            stream.write(hashlib.sha1(blocks[index]).digest())
    with files[DATA].open('r+b') as stream:
        stream.seek(385081726)
        require(struct.unpack('<I', stream.read(4))[0] == 5036, 'Metadata size mismatch')
        stream.seek(385081726)
        stream.write(struct.pack('<I', len(frontend)))
    actual = {name: digest(path) for name, path in files.items()}
    require(actual == EXPECTED, 'Overlay differs from the tested development resource set')
    (out / 'preparation.json').write_text(json.dumps({'recipe': 232, 'files': actual,
        'acceptance': 'Byte identity with development overlay only; no game acceptance inferred.'}, indent=2) + '\n')
    print('All 4 resource outputs match development overlay 232.')


if __name__ == '__main__':
    main()
