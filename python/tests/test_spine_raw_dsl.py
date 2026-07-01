# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""Byte-identical test for the direct-op DSL + executor.

Runs each DSL kernel (spine_raw_dsl_kernels.py — written with real op calls)
through SpineDSLExecutor and asserts the emitted MLIR matches the goldens
captured from the original marker codegen. Since the mv golden is itself
byte-identical to the raw_linalg that compiled & ran on K3 (ir_proton2), this
proves the new writing surface reproduces the K3-validated kernel exactly.

    python spine-triton/python/tests/test_spine_raw_dsl.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import spine_raw_dsl_kernels as K  # noqa: E402
from spine_raw.executor import SpineDSLExecutor  # noqa: E402

_GOLDEN_DIR = os.path.join(_HERE, "spine_raw_goldens")


def _emit(fn) -> str:
    body = SpineDSLExecutor(fn).generate()
    return "module {{\n{}\n}}\n".format(body)


def _check(name, fn):
    with open(os.path.join(_GOLDEN_DIR, f"{name}.mlir")) as f:
        golden = f.read()
    got = _emit(fn)
    if got != golden:
        g, o = golden.splitlines(), got.splitlines()
        for i, (a, b) in enumerate(zip(g, o)):
            if a != b:
                raise AssertionError(
                    f"{name}: first diff at line {i + 1}\n"
                    f"  golden: {a!r}\n  got   : {b!r}")
        raise AssertionError(
            f"{name}: length differs golden={len(g)} got={len(o)}")


def test_all_dsl_kernels():
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
    print(f"\n{len(K.ALL_KERNELS) - len(failures)}/{len(K.ALL_KERNELS)} "
          f"DSL kernels byte-identical to goldens")
    sys.exit(1 if failures else 0)
