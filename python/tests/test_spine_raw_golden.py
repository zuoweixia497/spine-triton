# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""Byte-identical regression test for the spine_raw codegen refactor.

Pins SpineMLIRCodeGenerator.make_linalg() output for every primitive against
goldens captured from the original f-string implementation. The typed-builder
rewrite must reproduce these byte-for-byte (CLAUDE.md: spine_raw v2 migration,
constraint A). Run directly or under pytest:

    python spine-triton/python/tests/test_spine_raw_golden.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import spine_raw_kernels as K  # noqa: E402

_GOLDEN_DIR = os.path.join(_HERE, "spine_raw_goldens")


def _check(name, fn):
    golden_path = os.path.join(_GOLDEN_DIR, f"{name}.mlir")
    with open(golden_path) as f:
        golden = f.read()
    got = fn.make_linalg()
    if got != golden:
        # Produce a compact first-diff report.
        g_lines, o_lines = golden.splitlines(), got.splitlines()
        for i, (gl, ol) in enumerate(zip(g_lines, o_lines)):
            if gl != ol:
                raise AssertionError(
                    f"{name}: first diff at line {i + 1}\n"
                    f"  golden: {gl!r}\n  got   : {ol!r}"
                )
        raise AssertionError(
            f"{name}: length differs golden={len(g_lines)} got={len(o_lines)} lines"
        )


def test_all_goldens():
    for name, fn in K.ALL_KERNELS.items():
        _check(name, fn)


if __name__ == "__main__":
    failures = []
    for name, fn in K.ALL_KERNELS.items():
        try:
            _check(name, fn)
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL {name}: {e}")
    print(f"\n{len(K.ALL_KERNELS) - len(failures)}/{len(K.ALL_KERNELS)} passed")
    sys.exit(1 if failures else 0)
