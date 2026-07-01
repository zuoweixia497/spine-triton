# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""Synthetic @spine_raw kernels exercising every spine_raw primitive.

Shared by the golden-capture driver and the byte-identical regression test
(test_spine_raw_golden.py). Each function clusters a few primitives so that the
full public surface of spine_raw is covered. These are NOT meant to run on
hardware — they exist purely so SpineMLIRCodeGenerator.make_linalg() output can
be pinned and diffed across the f-string → typed-builder refactor.

Import note: we load spine_raw from the SOURCE tree (spine-triton/language),
not the build copy, because make_linalg() is pure codegen (no JIT registry),
and the source is what the refactor edits. See CLAUDE.md rule 27 for why the
call() registry path is different (not exercised here).

NOTE: no `from __future__ import annotations` — In[...]/InOut[...] must evaluate
at def time so _parse_signature sees _TypedAnnotation objects, not strings.
"""
import os
import sys

_LANG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "language",
)
if _LANG_DIR not in sys.path:
    sys.path.insert(0, _LANG_DIR)

import spine_raw  # noqa: E402
from spine_raw import spine_raw as spine_raw_dec, In, InOut  # noqa: E402
import spine_raw as sr  # noqa: E402

_F16 = "memref<*xf16, #ptr.generic_space>"
_F32 = "memref<*xf32, #ptr.generic_space>"


# --- mv (the production kernel, full 2D / vfwmacc path) --------------------
@spine_raw_dec(name="linalg")
def k_mv_macc_block(
    B:   In[_F16],
    A:   In[_F16],
    col: In["index"],
    M:   In["index"],
    nk:  In["index"],
    C:   InOut[_F32],
):
    buf0 = sr.alloc_tcm_2d(32, 64, "f16")
    buf1 = sr.alloc_tcm_2d(32, 64, "f16")
    buf2 = sr.alloc_tcm_2d(32, 64, "f16")
    buf3 = sr.alloc_tcm_2d(32, 64, "f16")
    acc0 = sr.splat_2d(0.0, 1, 64, "f32")
    acc1 = sr.splat_2d(0.0, 1, 64, "f32")
    acc2 = sr.splat_2d(0.0, 1, 64, "f32")
    acc3 = sr.splat_2d(0.0, 1, 64, "f32")
    for kb in sr.range(nk):
        koff = kb * 32
        lhs = sr.view_2d(B, 1, 32, "f16", koff)
        sr.pack_2d_t_into(buf0, A, col,       32, 64, M, "f16", koff)
        sr.pack_2d_t_into(buf1, A, col + 64,  32, 64, M, "f16", koff)
        sr.pack_2d_t_into(buf2, A, col + 128, 32, 64, M, "f16", koff)
        sr.pack_2d_t_into(buf3, A, col + 192, 32, 64, M, "f16", koff)
        r0 = sr.load_2d(buf0, 32, 64, "f16")
        r1 = sr.load_2d(buf1, 32, 64, "f16")
        r2 = sr.load_2d(buf2, 32, 64, "f16")
        r3 = sr.load_2d(buf3, 32, 64, "f16")
        acc0 = sr.batch_macc(lhs, r0, acc0)
        acc1 = sr.batch_macc(lhs, r1, acc1)
        acc2 = sr.batch_macc(lhs, r2, acc2)
        acc3 = sr.batch_macc(lhs, r3, acc3)
    sr.store_2d_at(C, col,       1, 64, acc0)
    sr.store_2d_at(C, col + 64,  1, 64, acc1)
    sr.store_2d_at(C, col + 128, 1, 64, acc2)
    sr.store_2d_at(C, col + 192, 1, 64, acc3)


# --- proton scopes (proton_mark around clusters) ---------------------------
@spine_raw_dec(name="linalg")
def k_mv_proton(
    B:   In[_F16],
    A:   In[_F16],
    col: In["index"],
    M:   In["index"],
    nk:  In["index"],
    C:   InOut[_F32],
):
    sr.proton_mark("alloc", True)
    b0 = sr.alloc_tcm_2d(32, 64, "f16")
    sr.proton_mark("alloc", False)
    a0 = sr.splat_2d(0.0, 1, 64, "f32")
    for kb in sr.range(nk):
        koff = kb * 32
        lhs = sr.view_2d(B, 1, 32, "f16", koff)
        sr.proton_mark("pack", True)
        sr.pack_2d_t_into(b0, A, col, 32, 64, M, "f16", koff)
        sr.proton_mark("pack", False)
        r0 = sr.load_2d(b0, 32, 64, "f16")
        a0 = sr.batch_macc(lhs, r0, a0)
    sr.store_2d_at(C, col, 1, 64, a0)
    sr.free_tcm(b0)


# --- 1D vector cluster: splat / load_vec / extf / fma / reduce_add / store --
@spine_raw_dec(name="linalg")
def k_vec_1d(
    A: In[_F16],
    B: In[_F16],
    C: InOut[_F32],
    i: In["index"],
    n: In["index"],
):
    acc = sr.splat(0.0, shape=[32])
    for k in sr.range(n):
        a = sr.load_vec(A, i, 32, "f16")
        b = sr.load_vec(B, i, 32, "f16")
        af = sr.extf(a, "f32")
        bf = sr.extf(b, "f32")
        acc = sr.fma(af, bf, acc)
    s = sr.reduce_add(acc)
    sr.store_scalar(C, i, s)
    sr.store_vec(C, i, acc)


# --- matmul / pad_vec / extract_elem / load_tile ---------------------------
@spine_raw_dec(name="linalg")
def k_matmul(
    A:  In[_F16],
    C:  InOut[_F32],
    i:  In["index"],
    rb: In["index"],
    cb: In["index"],
):
    lhs = sr.load_vec(A, i, 32, "f16")
    rhs = sr.load_vec(A, i, 2048, "f16")
    accv = sr.splat(0.0, shape=[64])
    out = sr.matmul(lhs, rhs, accv, 1, 64, 32)
    e0 = sr.extract_elem(out, 0)
    ei = sr.extract_elem(out, i)
    sr.store_scalar(C, i, e0)
    sr.store_scalar(C, rb, ei)
    padded = sr.pad_vec(lhs, 2048)
    out2 = sr.matmul(lhs, padded, accv, 1, 64, 32)
    sr.store_vec(C, cb, out2)
    tile = sr.load_tile(A, rb, cb, 32, 1, 32, "f16")
    sr.store_vec(C, i, tile)


# --- 2D read variants: view_2d(no off) / load_2d_at / load_2d_t / pack_2d_t -
@spine_raw_dec(name="linalg")
def k_2d_reads(
    A:  In[_F16],
    C:  InOut[_F32],
    rb: In["index"],
    off: In["index"],
    M:  In["index"],
    co: In["index"],
):
    v0 = sr.view_2d(A, 1, 32, "f16")
    sr.store_2d(C, 1, 32, sr.load_2d(v0, 1, 32, "f16"))
    la = sr.load_2d_at(A, off, 64, 32, "f16")
    sr.store_2d_at(C, off, 64, 32, la)
    lt_static = sr.load_2d_t(A, rb, 32, 64, 128, "f16")
    sr.store_2d(C, 32, 64, lt_static)
    lt_dyn = sr.load_2d_t(A, rb, 32, 64, M, "f16", co)
    sr.store_2d(C, 32, 64, lt_dyn)
    pk = sr.pack_2d_t(A, rb, 32, 64, M, "f16", co)
    sr.store_2d(C, 32, 64, pk)


# Registry consumed by the capture driver and the regression test.
ALL_KERNELS = {
    "mv_macc_block": k_mv_macc_block,
    "mv_proton": k_mv_proton,
    "vec_1d": k_vec_1d,
    "matmul": k_matmul,
    "2d_reads": k_2d_reads,
}
