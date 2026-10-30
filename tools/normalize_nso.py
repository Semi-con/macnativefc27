#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (c) 2026 macnativefc27 contributors
"""Standalone offline NSO normalizer. Run in a separate Python environment.

This program uses the user's separately installed zbic package; no decoder or
game payload is included here. It does not import or link the Rust runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def normalize(source, destination):
    original = source.read_bytes()
    if len(original) < 256 or original[:4] != b'NSO0':
        raise ValueError('Expected an NSO0 file')
    flags = struct.unpack_from('<I', original, 12)[0]
    segments = [struct.unpack_from('<III', original, 0x10 + 16*i) for i in range(3)]
    compressed = struct.unpack_from('<III', original, 0x60)
    first = min(s[0] for s in segments)
    if not 256 <= first <= len(original):
        raise ValueError('Invalid NSO header extent')
    result = bytearray(original[:first])
    struct.pack_into('<I', result, 12, flags & ~0x87)
    checks = []
    for i, (offset, memory, size) in enumerate(segments):
        encoded_size = compressed[i] if flags & (1 << i) else size
        if not 0 < size <= 256*1024*1024 or offset < first or offset + encoded_size > len(original):
            raise ValueError('NSO segment exceeds bounds')
        encoded = original[offset:offset + encoded_size]
        if flags & (1 << i):
            if flags & 0x80:
                import zbic
                if encoded[:4] != b'ZBIC':
                    raise ValueError('Missing ZBIC magic')
                decoded, codec = zbic.decompress(encoded), 'ZBIC'
            else:
                import lz4.block
                decoded, codec = lz4.block.decompress(encoded, uncompressed_size=size), 'LZ4'
        else:
            decoded, codec = encoded, 'none'
        digest = hashlib.sha256(decoded).digest()
        if len(decoded) != size or not flags & (8 << i) or digest != original[0xa0+32*i:0xc0+32*i]:
            raise ValueError(f'{source.name} segment {i}: original header hash mismatch')
        struct.pack_into('<I', result, 0x10+16*i, len(result))
        struct.pack_into('<I', result, 0x60+4*i, size)
        result.extend(decoded)
        checks.append({'segment': i, 'codec': codec, 'sha256': digest.hex(), 'bytes': size})
    with destination.open('xb') as out:
        out.write(result)
    return {'module': source.name, 'source_sha256': hashlib.sha256(original).hexdigest(),
            'normalized_sha256': hashlib.sha256(result).hexdigest(), 'segments': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = [normalize(args.source / name, args.out / name) for name in ('main', 'rtld', 'sdk')]
    (args.out / 'normalization.json').write_text(json.dumps(records, indent=2) + '\n')
    print('Normalized 3 modules; all 9 original segment hashes verified.')


if __name__ == '__main__':
    main()
