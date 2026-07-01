# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""spine_raw direct-op DSL — the FlagTree-style writing surface.

A kernel body calls these as real Python functions (no `sr.` markers, no
hidden codegen routing):

    @spine_kernel
    def mv(B: In["memref<*xf16, #ptr.generic_space>"], ..., C: InOut[...]):
        buf0 = alloc_tcm_2d(32, 64, "f16")
        acc0 = splat_2d(0.0, 1, 64, "f32")
        for kb in srange(nk):
            koff = kb * 32
            lhs = view_2d(B, 1, 32, "f16", koff)
            pack_2d_t_into(buf0, A, col, 32, 64, M, "f16", koff)
            r0 = load_2d(buf0, 32, 64, "f16")
            acc0 = batch_macc(lhs, r0, acc0)
        store_2d_at(C, col, 1, 64, acc0)

The executor (executor.py) runs the body with an implicit Builder bound in
ir.CTX and the current assignment target pushed as the result-name hint, so
each function just forwards to the typed constructors in ops.py. Output matches
the original codegen byte-for-byte (the names come from the Python variables).

These functions only run inside an executor-driven kernel; calling one outside
raises because ir.CTX.builder is None.
"""
from . import ops
from .ir import CTX, Type, Value
from .types import In, InOut  # re-exported for kernel signatures  # noqa: F401


def _b():
    if CTX.builder is None:
        raise RuntimeError(
            "spine_raw.dsl ops must run inside a @spine_kernel body "
            "(no Builder bound in ir.CTX)"
        )
    return CTX.builder


def _hint() -> str:
    return CTX.take_hint()


def _val(x, dtype: str = "f32") -> Value:
    """Coerce a Python scalar literal to a constant Value (else pass through)."""
    if isinstance(x, Value):
        return x
    if isinstance(x, float):
        return _b().cfloat(x)  # original always pools floats as f32
    if isinstance(x, int):
        return _b().cint(x)
    raise TypeError(f"cannot use {x!r} as an operand")


# -- loop marker ------------------------------------------------------------

class srange:
    """for kb in srange(n): ... → scf.for with auto iter_args.

    A marker consumed by the executor's visit_For; never actually iterated in
    Python (the trip count is a runtime Value)."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        raise RuntimeError("srange is interpreted by the spine_raw executor, "
                           "not iterated in Python")


# -- 1-D vector / arith -----------------------------------------------------

def splat(val, shape):
    assert len(shape) == 1, "splat only supports 1D shape"
    return ops.splat(_b(), _val(val), shape[0], _hint())


def load_vec(ptr, idx, n, dtype="f32", in_bounds=True):
    return ops.load_vec(_b(), ptr, _val(idx), n, dtype, in_bounds, _hint())


def extf(v, dtype="f32"):
    return ops.extf(_b(), v, dtype, _hint())


def fma(a, b, c):
    return ops.fma(_b(), a, b, c, _hint())


def reduce_add(v):
    return ops.reduce_add(_b(), v, _hint())


def matmul(lhs, rhs, acc, m, n, k):
    return ops.matmul(_b(), lhs, rhs, acc, m, n, k, _hint())


def load_tile(ptr, row_base, col_base, row_stride, m, k, dtype="f16"):
    return ops.load_tile(_b(), ptr, row_base, col_base, row_stride, m, k, dtype, _hint())


def pad_vec(v, total):
    return ops.pad_vec(_b(), v, total, _hint())


def extract_elem(v, idx):
    return ops.extract_elem(_b(), v, idx, _hint())


# -- 2-D / vfwmacc path -----------------------------------------------------

def splat_2d(val, rows, cols, dtype="f32"):
    return ops.splat_2d(_b(), _val(val), rows, cols, dtype, _hint())


def batch_macc(lhs, rhs, acc):
    return ops.batch_macc(_b(), lhs, rhs, acc, _hint())


def view_2d(ptr, rows, cols, dtype="f16", off=None):
    return ops.view_2d(_b(), ptr, rows, cols, dtype, off, _hint())


def load_2d(ptr, rows, cols, dtype="f16"):
    return ops.load_2d(_b(), ptr, rows, cols, dtype, _hint())


def load_2d_at(ptr, off, rows, cols, dtype="f16"):
    return ops.load_2d_at(_b(), ptr, off, rows, cols, dtype, _hint())


def load_2d_t(ptr, row_base, k, nb, m, dtype="f16", col_off=None):
    return ops.load_2d_t(_b(), ptr, row_base, k, nb, m, dtype, col_off, _hint())


def pack_2d_t(ptr, row_base, k, nb, m, dtype="f16", col_off=None):
    return ops.pack_2d_t(_b(), ptr, row_base, k, nb, m, dtype, col_off, _hint())


def alloc_tcm_2d(k, nb, dtype="f16"):
    return ops.alloc_tcm_2d(_b(), k, nb, dtype, _hint())


def pack_2d_t_into(buf, ptr, row_base, k, nb, m, dtype="f16", col_off=None):
    ops.pack_2d_t_into(_b(), buf, ptr, row_base, k, nb, m, dtype, col_off)


def free_tcm(buf):
    ops.free_tcm(_b(), buf)


def proton_mark(name, is_start):
    ops.proton_mark(_b(), name, is_start)


# -- stores -----------------------------------------------------------------

def store_vec(ptr, idx, vec):
    ops.store_vec(_b(), ptr, _val(idx), vec)


def store_scalar(ptr, idx, val):
    ops.store_scalar(_b(), ptr, _val(idx), val)


def store_2d(ptr, rows, cols, vec):
    ops.store_2d(_b(), ptr, rows, cols, vec)


def store_2d_at(ptr, off, rows, cols, vec):
    ops.store_2d_at(_b(), ptr, off, rows, cols, vec)
