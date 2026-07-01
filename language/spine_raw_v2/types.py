# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""spine_raw_v2 type annotations — In, InOut for raw kernel parameters."""

from __future__ import annotations
from mlir import ir


class In:
    """Read-only parameter annotation."""
    def __init__(self, mlir_type_str: str | type):
        self.mlir_type = _resolve_type(mlir_type_str)

    def __repr__(self):
        return f"In[{self.mlir_type}]"


class InOut:
    """Read-write parameter annotation."""
    def __init__(self, mlir_type_str: str | type):
        self.mlir_type = _resolve_type(mlir_type_str)

    def __repr__(self):
        return f"InOut[{self.mlir_type}]"


def _resolve_type(t) -> ir.Type:
    if isinstance(t, ir.Type):
        return t
    if isinstance(t, str):
        if t == "index":
            return ir.IndexType.get()
        if t == "i32":
            return ir.IntegerType.get_signless(32)
        if t == "i64":
            return ir.IntegerType.get_signless(64)
        if t == "f16":
            return ir.F16Type.get()
        if t == "f32":
            return ir.F32Type.get()
        # memref type strings: e.g. "memref<*xf16, #ptr.generic_space>"
        if t.startswith("memref<"):
            return ir.Type.parse(t)
    raise ValueError(f"cannot resolve type: {t!r}")
