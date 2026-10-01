#!/usr/bin/env python3
"""Expose framework header declarations in ParaS's main rewrite buffer.

Only quoted includes resolved under csrc are expanded. SYCL, Torch, and
system headers remain ordinary includes. Each expanded header keeps an
include guard so repeated and conditional includes retain their semantics.
"""

import argparse
import hashlib
from pathlib import Path
import re


INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"\s*(?://.*)?$')
PRAGMA_ONCE = re.compile(r'^\s*#\s*pragma\s+once\s*$')


def flatten(source: Path, root: Path) -> str:
    def expand(path: Path, active: tuple[Path, ...]) -> str:
        if path in active:
            raise ValueError(f"cyclic framework include: {path}")
        output = []
        for line in path.read_text().splitlines(keepends=True):
            match = INCLUDE.match(line)
            if match:
                name = match.group(1)
                candidates = [(path.parent / name).resolve(), (root / name).resolve()]
                header = next((p for p in candidates if p.is_file() and p.is_relative_to(root)), None)
                if header is not None:
                    relative = header.relative_to(root).as_posix()
                    guard = "PTSYCL_FLATTENED_" + hashlib.sha256(relative.encode()).hexdigest()[:16].upper()
                    output.append(f"\n// Framework header: {relative}\n#ifndef {guard}\n#define {guard}\n")
                    output.append(expand(header, active + (path,)))
                    output.append(f"\n#endif // {guard}\n")
                    continue
            if not PRAGMA_ONCE.match(line):
                output.append(line)
        return "".join(output)

    # Parse SYCL's _Float16 declarations before Torch's CUDA compatibility
    # headers, which can define _Float16 as __half. Include guards keep later
    # source includes harmless and preserve the compiler's native half type.
    return "#include <sycl/sycl.hpp>\n" + expand(source.resolve(), ())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(flatten(args.source, args.root.resolve()))


if __name__ == "__main__":
    main()
