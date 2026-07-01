# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""The synthetic kernels rewritten in the direct-op DSL (FlagTree-style).

Same logic as spine_raw_kernels.py but using real function calls
(alloc_tcm_2d(...), batch_macc(...), for kb in srange(...)) instead of sr.*
markers, run through SpineDSLExecutor. Names match the marker goldens so the
executor's output can be diffed byte-for-byte (tests/test_spine_raw_dsl.py).
"""
import os
import sys

_LANG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "language",
)
if _LANG_DIR not in sys.path:
    sys.path.insert(0, _LANG_DIR)

from spine_raw.dsl import (  # noqa: E402
    In, InOut, srange,
    splat, load_vec, store_vec, store_scalar, extf, fma, reduce_add, matmul,
    load_tile, pad_vec, extract_elem,
    splat_2d, batch_macc, view_2d, load_2d, load_2d_at, load_2d_t, pack_2d_t,
    alloc_tcm_2d, pack_2d_t_into, free_tcm, proton_mark, store_2d, store_2d_at,
)

_F16 = "memref<*xf16, #ptr.generic_space>"
_F32 = "memref<*xf32, #ptr.generic_space>"


def k_mv_macc_block(B: In[_F16], A: In[_F16], col: In["index"],
                    M: In["index"], nk: In["index"], C: InOut[_F32]):
    buf0 = alloc_tcm_2d(32, 64, "f16")
    buf1 = alloc_tcm_2d(32, 64, "f16")
    buf2 = alloc_tcm_2d(32, 64, "f16")
    buf3 = alloc_tcm_2d(32, 64, "f16")
    acc0 = splat_2d(0.0, 1, 64, "f32")
    acc1 = splat_2d(0.0, 1, 64, "f32")
    acc2 = splat_2d(0.0, 1, 64, "f32")
    acc3 = splat_2d(0.0, 1, 64, "f32")
    for kb in srange(nk):
        koff = kb * 32
        lhs = view_2d(B, 1, 32, "f16", koff)
        pack_2d_t_into(buf0, A, col,       32, 64, M, "f16", koff)
        pack_2d_t_into(buf1, A, col + 64,  32, 64, M, "f16", koff)
        pack_2d_t_into(buf2, A, col + 128, 32, 64, M, "f16", koff)
        pack_2d_t_into(buf3, A, col + 192, 32, 64, M, "f16", koff)
        r0 = load_2d(buf0, 32, 64, "f16")
        r1 = load_2d(buf1, 32, 64, "f16")
        r2 = load_2d(buf2, 32, 64, "f16")
        r3 = load_2d(buf3, 32, 64, "f16")
        acc0 = batch_macc(lhs, r0, acc0)
        acc1 = batch_macc(lhs, r1, acc1)
        acc2 = batch_macc(lhs, r2, acc2)
        acc3 = batch_macc(lhs, r3, acc3)
    store_2d_at(C, col,       1, 64, acc0)
    store_2d_at(C, col + 64,  1, 64, acc1)
    store_2d_at(C, col + 128, 1, 64, acc2)
    store_2d_at(C, col + 192, 1, 64, acc3)


def k_mv_proton(B: In[_F16], A: In[_F16], col: In["index"],
                M: In["index"], nk: In["index"], C: InOut[_F32]):
    proton_mark("alloc", True)
    b0 = alloc_tcm_2d(32, 64, "f16")
    proton_mark("alloc", False)
    a0 = splat_2d(0.0, 1, 64, "f32")
    for kb in srange(nk):
        koff = kb * 32
        lhs = view_2d(B, 1, 32, "f16", koff)
        proton_mark("pack", True)
        pack_2d_t_into(b0, A, col, 32, 64, M, "f16", koff)
        proton_mark("pack", False)
        r0 = load_2d(b0, 32, 64, "f16")
        a0 = batch_macc(lhs, r0, a0)
    store_2d_at(C, col, 1, 64, a0)
    free_tcm(b0)


def k_vec_1d(A: In[_F16], B: In[_F16], C: InOut[_F32], i: In["index"], n: In["index"]):
    acc = splat(0.0, shape=[32])
    for k in srange(n):
        a = load_vec(A, i, 32, "f16")
        b = load_vec(B, i, 32, "f16")
        af = extf(a, "f32")
        bf = extf(b, "f32")
        acc = fma(af, bf, acc)
    s = reduce_add(acc)
    store_scalar(C, i, s)
    store_vec(C, i, acc)


def k_matmul(A: In[_F16], C: InOut[_F32], i: In["index"],
             rb: In["index"], cb: In["index"]):
    lhs = load_vec(A, i, 32, "f16")
    rhs = load_vec(A, i, 2048, "f16")
    accv = splat(0.0, shape=[64])
    out = matmul(lhs, rhs, accv, 1, 64, 32)
    e0 = extract_elem(out, 0)
    ei = extract_elem(out, i)
    store_scalar(C, i, e0)
    store_scalar(C, rb, ei)
    padded = pad_vec(lhs, 2048)
    out2 = matmul(lhs, padded, accv, 1, 64, 32)
    store_vec(C, cb, out2)
    tile = load_tile(A, rb, cb, 32, 1, 32, "f16")
    store_vec(C, i, tile)


def k_2d_reads(A: In[_F16], C: InOut[_F32], rb: In["index"],
               off: In["index"], M: In["index"], co: In["index"]):
    v0 = view_2d(A, 1, 32, "f16")
    store_2d(C, 1, 32, load_2d(v0, 1, 32, "f16"))
    la = load_2d_at(A, off, 64, 32, "f16")
    store_2d_at(C, off, 64, 32, la)
    lt_static = load_2d_t(A, rb, 32, 64, 128, "f16")
    store_2d(C, 32, 64, lt_static)
    lt_dyn = load_2d_t(A, rb, 32, 64, M, "f16", co)
    store_2d(C, 32, 64, lt_dyn)
    pk = pack_2d_t(A, rb, 32, 64, M, "f16", co)
    store_2d(C, 32, 64, pk)


ALL_KERNELS = {
    "mv_macc_block": k_mv_macc_block,
    "mv_proton": k_mv_proton,
    "vec_1d": k_vec_1d,
    "matmul": k_matmul,
    "2d_reads": k_2d_reads,
}
