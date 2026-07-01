# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""spine_raw op constructors — the typed layer over ir.Builder.

Each function takes a Builder plus already-resolved arguments (ir.Value for
runtime SSA values, plain int/str for static shape/dtype params) and emits the
corresponding MLIR op, returning an ir.Value (or None for side-effecting
stores). This replaces the original SpineMLIRCodeGenerator._gen_* f-string
blobs: SSA/constant/indent bookkeeping now lives in the Builder, types flow as
ir.Type, and shape/dtype constraints are checked here at build time.

Output is byte-identical to the original implementation (verified against
tests/spine_raw_goldens/*), so the sequence of Builder.new()/cint()/cfloat()/
emit() calls within each function is deliberately preserved.
"""
from .ir import Builder, Type, Value


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ranked(b: Builder, v: Value) -> Value:
    """Cast an unranked memref<*xT...> to ranked memref<?xT...>; pass through
    anything already ranked. Mirrors the old _ranked_cast."""
    if v.type.is_unranked_memref:
        rt = v.type.to_ranked()
        return b.define("ranked", rt, f"memref.cast {v.ssa} : {v.type} to {rt}")
    return v


def _strided_2d_type(rows, cols, dtype: str, space: str) -> Type:
    sp = f", {space}" if space else ""
    return Type(f"memref<{rows}x{cols}x{dtype}, strided<[{cols}, 1]>{sp}>")


def _is_static(x) -> bool:
    """A dimension/stride argument is static when given as a Python int."""
    return isinstance(x, int)


# ---------------------------------------------------------------------------
# 1-D vector / arith
# ---------------------------------------------------------------------------

def splat(b: Builder, val: Value, n: int, hint: str) -> Value:
    vt = Type.vector([n], "f32")
    return b.define(hint or "splat", vt, f"vector.broadcast {val.ssa} : f32 to {vt}")


def load_vec(b: Builder, ptr: Value, idx: Value, n: int, dtype: str,
             in_bounds: bool, hint: str) -> Value:
    assert ptr.type.is_memref, f"load_vec needs a memref, got {ptr.type}"
    src = _ranked(b, ptr)
    vt = Type.vector([n], dtype)
    res = b.new(hint or "vec")
    pad = b.cfloat(0.0, dtype)
    ib = "true" if in_bounds else "false"
    b.emit(f"{res} = vector.transfer_read {src.ssa}[{idx.ssa}], {pad.ssa}"
           f" {{in_bounds = [{ib}]}} : {src.type}, {vt}")
    return Value(res, vt)


def extf(b: Builder, v: Value, dst_dtype: str, hint: str) -> Value:
    assert v.type.is_vector, f"extf needs a vector, got {v.type}"
    dst = Type.vector([v.type.vec_n], dst_dtype)
    return b.define(hint or "extf", dst, f"arith.extf {v.ssa} : {v.type} to {dst}")


def fma(b: Builder, a: Value, bb: Value, c: Value, hint: str) -> Value:
    assert a.type == c.type, f"fma: a and acc types must match: {a.type} vs {c.type}"
    return b.define(hint or "fma", a.type,
                    f"math.fma {a.ssa}, {bb.ssa}, {c.ssa} : {a.type}")


def reduce_add(b: Builder, v: Value, hint: str) -> Value:
    elem = Type.scalar(v.type.vec_rest)
    return b.define(hint or "rsum", elem,
                    f"vector.reduction <add>, {v.ssa} : {v.type} into {elem}")


def matmul(b: Builder, lhs: Value, rhs: Value, acc: Value,
           m: int, n: int, k: int, hint: str) -> Value:
    # out[m,n] = acc[m,n] + sum_k lhs[m,k] * rhs[n,k]; generic op form so the
    # raw_linalg text parses in tools where vector_ext is unregistered.
    rhs_type = Type.vector([n * k], lhs.type.vec_rest)
    return b.define(
        hint or "mm", acc.type,
        f'"vector_ext.matmul"({lhs.ssa}, {rhs.ssa}, {acc.ssa})'
        f' <{{m = {m} : i64, n = {n} : i64, k = {k} : i64}}>'
        f' : ({lhs.type}, {rhs_type}, {acc.type}) -> {acc.type}')


def load_tile(b: Builder, ptr: Value, row_base: Value, col_base: Value,
              row_stride: int, m: int, k: int, dtype: str, hint: str) -> Value:
    # Read an M×K row-major tile via a 2D transfer_read, then flatten.
    ranked1d = ptr.type.to_ranked()
    space = ptr.type.memref_space
    sp = f", {space}" if space else ""
    ranked2d = Type(f"memref<?x{row_stride}x{dtype}, strided<[{row_stride}, 1], "
                    f"offset: ?>{sp}>")
    c0 = b.cint(0)
    r1 = b.define("ranked", ranked1d, f"memref.cast {ptr.ssa} : {ptr.type} to {ranked1d}")
    view = b.new("view2d")
    size1d = b.define("sz1d", Type.index(), f"memref.dim {r1.ssa}, {c0.ssa} : {ranked1d}")
    b.emit(f"{view} = memref.reinterpret_cast {r1.ssa} to "
           f"offset: [0], sizes: [{size1d.ssa}, {row_stride}], "
           f"strides: [{row_stride}, 1]"
           f" : {ranked1d} to {ranked2d}")
    tile2d = Type(f"vector<{m}x{k}x{dtype}>")
    flat = Type.vector([m * k], dtype)
    pad = b.cfloat(0.0, dtype)
    res2d = b.new(hint or "tile2d")
    b.emit(f"{res2d} = vector.transfer_read {view}[{row_base.ssa}, {col_base.ssa}], {pad.ssa}"
           f" {{in_bounds = [true, true]}} : {ranked2d}, {tile2d}")
    res = b.new(hint or "tile")
    b.emit(f"{res} = vector.shape_cast {res2d} : {tile2d} to {flat}")
    return Value(res, flat)


def pad_vec(b: Builder, v: Value, total: int, hint: str) -> Value:
    dtype = v.type.vec_rest
    full = Type.vector([total], dtype)
    zero = b.cfloat(0.0, dtype)
    base = b.define(hint or "pad", full, f"vector.broadcast {zero.ssa} : {dtype} to {full}")
    return b.define(hint or "pad", full,
                    f"vector.insert_strided_slice {v.ssa}, {base.ssa}"
                    f" {{offsets = [0], strides = [1]}} : {v.type} into {full}")


def extract_elem(b: Builder, v: Value, idx, hint: str) -> Value:
    # idx: int (static -> vector.extract) or Value (dynamic -> extractelement).
    dtype = Type.scalar(v.type.vec_rest)
    if _is_static(idx):
        return b.define(hint or "elem", dtype,
                        f"vector.extract {v.ssa}[{idx}] : {dtype} from {v.type}")
    assert idx.type.is_index, f"extract_elem dynamic index must be index, got {idx.type}"
    return b.define(hint or "elem", dtype,
                    f"vector.extractelement {v.ssa}[{idx.ssa} : index] : {v.type}")


# ---------------------------------------------------------------------------
# 2-D / vfwmacc path
# ---------------------------------------------------------------------------

def splat_2d(b: Builder, val: Value, rows: int, cols: int, dtype: str, hint: str) -> Value:
    vt = Type(f"vector<{rows}x{cols}x{dtype}>")
    return b.define(hint or "splat2d", vt,
                    f"vector.broadcast {val.ssa} : {dtype} to {vt}")


def batch_macc(b: Builder, lhs: Value, rhs: Value, acc: Value, hint: str) -> Value:
    # acc[m,n] = sum_k lhs[m,k]*rhs[k,n]; generic op form (vector_ext may be
    # unregistered in the parsing tool). n (output cols) must be a multiple of
    # 64 for the K3 vfwmacc lowering — checked here at build time.
    n_cols = acc.type.text  # acc is vector<m x n x dtype>
    import re as _re
    mm = _re.match(r"vector<\d+x(\d+)x", n_cols)
    if mm:
        n = int(mm.group(1))
        assert n % 64 == 0 and n >= 64, \
            f"batch_macc output cols n={n} must be a multiple of 64 (K3 vfwmacc)"
    return b.define(
        hint or "bmacc", acc.type,
        f'"vector_ext.batch_macc"({lhs.ssa}, {rhs.ssa}, {acc.ssa})'
        f' : ({lhs.type}, {rhs.type}, {acc.type}) -> {acc.type}')


def view_2d(b: Builder, ptr: Value, rows: int, cols: int, dtype: str,
            off: Value, hint: str) -> Value:
    # 2D strided view; off is an optional dynamic element offset (or None).
    space = ptr.type.memref_space
    src = _ranked(b, ptr)
    if off is not None:
        sp = f", {space}" if space else ""
        out_type = Type(f"memref<{rows}x{cols}x{dtype}, strided<[{cols}, 1], "
                        f"offset: ?>{sp}>")
        off_part = f"[{off.ssa}]"
    else:
        out_type = _strided_2d_type(rows, cols, dtype, space)
        off_part = "[0]"
    return b.define(hint or "view2d", out_type,
                    f"memref.reinterpret_cast {src.ssa} to "
                    f"offset: {off_part}, sizes: [{rows}, {cols}], strides: [{cols}, 1]"
                    f" : {src.type} to {out_type}")


def load_2d(b: Builder, ptr: Value, rows: int, cols: int, dtype: str, hint: str) -> Value:
    space = ptr.type.memref_space
    src = _ranked(b, ptr)
    mtype = _strided_2d_type(rows, cols, dtype, space)
    view = b.define("view2d", mtype,
                    f"memref.reinterpret_cast {src.ssa} to "
                    f"offset: [0], sizes: [{rows}, {cols}], strides: [{cols}, 1]"
                    f" : {src.type} to {mtype}")
    c0 = b.cint(0)
    pad = b.cfloat(0.0, dtype)
    vt = Type(f"vector<{rows}x{cols}x{dtype}>")
    return b.define(hint or "ld2d", vt,
                    f"vector.transfer_read {view.ssa}[{c0.ssa}, {c0.ssa}], {pad.ssa}"
                    f" {{in_bounds = [true, true]}} : {mtype}, {vt}")


def load_2d_at(b: Builder, ptr: Value, off: Value, rows: int, cols: int,
               dtype: str, hint: str) -> Value:
    space = ptr.type.memref_space
    src = _ranked(b, ptr)
    rows_c = b.cint(rows)
    boff = b.define("boff", Type.index(), f"arith.muli {off.ssa}, {rows_c.ssa} : index")
    sp = f", {space}" if space else ""
    mtype = Type(f"memref<{rows}x{cols}x{dtype}, strided<[{cols}, 1], offset: ?>{sp}>")
    view = b.define("view2d", mtype,
                    f"memref.reinterpret_cast {src.ssa} to "
                    f"offset: [{boff.ssa}], sizes: [{rows}, {cols}], strides: [{cols}, 1]"
                    f" : {src.type} to {mtype}")
    c0 = b.cint(0)
    pad = b.cfloat(0.0, dtype)
    vt = Type(f"vector<{rows}x{cols}x{dtype}>")
    return b.define(hint or "ld2d", vt,
                    f"vector.transfer_read {view.ssa}[{c0.ssa}, {c0.ssa}], {pad.ssa}"
                    f" {{in_bounds = [true, true]}} : {mtype}, {vt}")


def _t_offset(b: Builder, row_base: Value, m, col_off: Value):
    """Shared offset computation for the transposed 2D reads.

    m: int (static stride) or Value (dynamic). Returns (boff Value, m_ssa,
    col_stride_text, stride_operand_text)."""
    if _is_static(m):
        m_ssa = b.cint(m).ssa
        col_stride = str(m)
        stride_op = str(m)
    else:
        m_ssa = m.ssa
        col_stride = "?"
        stride_op = m.ssa
    boff = b.define("boff", Type.index(), f"arith.muli {row_base.ssa}, {m_ssa} : index")
    if col_off is not None:
        boff = b.define("boff", Type.index(),
                        f"arith.addi {boff.ssa}, {col_off.ssa} : index")
    return boff, col_stride, stride_op


def load_2d_t(b: Builder, ptr: Value, row_base: Value, k: int, nb: int, m,
              dtype: str, col_off: Value, hint: str) -> Value:
    # Transposed read from row-major A: result[k,c] = A[row_base+c, col_off+k].
    # View offset = row_base*M + col_off, sizes=[K,NB], strides=[1, M].
    space = ptr.type.memref_space
    src = _ranked(b, ptr)
    boff, col_stride, stride_op = _t_offset(b, row_base, m, col_off)
    sp = f", {space}" if space else ""
    mtype = Type(f"memref<{k}x{nb}x{dtype}, strided<[1, {col_stride}], offset: ?>{sp}>")
    view = b.define("viewT", mtype,
                    f"memref.reinterpret_cast {src.ssa} to "
                    f"offset: [{boff.ssa}], sizes: [{k}, {nb}], strides: [1, {stride_op}]"
                    f" : {src.type} to {mtype}")
    c0 = b.cint(0)
    pad = b.cfloat(0.0, dtype)
    vt = Type(f"vector<{k}x{nb}x{dtype}>")
    return b.define(hint or "ldT", vt,
                    f"vector.transfer_read {view.ssa}[{c0.ssa}, {c0.ssa}], {pad.ssa}"
                    f" {{in_bounds = [true, true]}} : {mtype}, {vt}")


def _pack_view(b: Builder, ptr: Value, row_base: Value, k: int, nb: int, m,
               dtype: str, col_off: Value):
    """Emit the row-major A view for a transposed pack. Returns (a_view, row_type).

    Shared by pack_2d_t and pack_2d_t_into. The destination buffer is NOT
    touched here so callers control where its alloca lands (the original
    pack_2d_t allocates buf *after* this view)."""
    space = ptr.type.memref_space
    src = _ranked(b, ptr)
    if _is_static(m):
        m_ssa = b.cint(m).ssa
        row_stride = str(m)
        stride_op = str(m)
    else:
        m_ssa = m.ssa
        row_stride = "?"
        stride_op = m.ssa
    boff = b.define("boff", Type.index(), f"arith.muli {row_base.ssa}, {m_ssa} : index")
    if col_off is not None:
        boff = b.define("boff", Type.index(),
                        f"arith.addi {boff.ssa}, {col_off.ssa} : index")
    sp = f", {space}" if space else ""
    row_type = Type(f"memref<{nb}x{k}x{dtype}, strided<[{row_stride}, 1], offset: ?>{sp}>")
    a_view = b.define("Aview", row_type,
                      f"memref.reinterpret_cast {src.ssa} to "
                      f"offset: [{boff.ssa}], sizes: [{nb}, {k}], strides: [{stride_op}, 1]"
                      f" : {src.type} to {row_type}")
    return a_view, row_type


def _emit_transpose(b: Builder, a_view: Value, row_type: Type, dtype: str,
                    buf_ssa: str, buf_type: Type) -> None:
    """Emit the linalg.generic that transposes a_view (NB×K) into buf (K×NB)."""
    d0_map = "affine_map<(d0, d1) -> (d1, d0)>"
    d1_map = "affine_map<(d0, d1) -> (d0, d1)>"
    b.emit(f'linalg.generic {{indexing_maps = [{d0_map}, {d1_map}], '
           f'iterator_types = ["parallel", "parallel"]'
           f'}} ins({a_view.ssa} : {row_type}) outs({buf_ssa} : {buf_type}) {{')
    b.emit(f'^bb0(%a: {dtype}, %_: {dtype}):')
    b.emit(f'  linalg.yield %a : {dtype}')
    b.emit('}')


def pack_2d_t(b: Builder, ptr: Value, row_base: Value, k: int, nb: int, m,
              dtype: str, col_off: Value, hint: str) -> Value:
    # Like load_2d_t but reads A row-major (unit inner stride) then transposes
    # via a stack buffer — no extra DDR bandwidth. buf alloca follows the view.
    a_view, row_type = _pack_view(b, ptr, row_base, k, nb, m, dtype, col_off)
    buf_type = Type(f"memref<{k}x{nb}x{dtype}>")
    buf = b.define("buf", buf_type, f"memref.alloca() {{alignment = 64 : i64}} : {buf_type}")
    _emit_transpose(b, a_view, row_type, dtype, buf.ssa, buf_type)
    c0 = b.cint(0)
    pad = b.cfloat(0.0, dtype)
    vt = Type(f"vector<{k}x{nb}x{dtype}>")
    return b.define(hint or "pkT", vt,
                    f"vector.transfer_read {buf.ssa}[{c0.ssa}, {c0.ssa}], {pad.ssa}"
                    f" {{in_bounds = [true, true]}} : {buf_type}, {vt}")


def alloc_tcm_2d(b: Builder, k: int, nb: int, dtype: str, hint: str) -> Value:
    mtype = Type(f"memref<{k}x{nb}x{dtype}>")
    return b.define(hint or "tcmbuf", mtype,
                    f"memref.alloc() {{alignment = 64 : i64}} : {mtype}")


def pack_2d_t_into(b: Builder, buf: Value, ptr: Value, row_base: Value,
                   k: int, nb: int, m, dtype: str, col_off: Value) -> None:
    # Pack A transposed into an existing buf (no alloca). buf is typically a TCM
    # buffer from alloc_tcm_2d.
    a_view, row_type = _pack_view(b, ptr, row_base, k, nb, m, dtype, col_off)
    buf_type = Type(f"memref<{k}x{nb}x{dtype}>")
    _emit_transpose(b, a_view, row_type, dtype, buf.ssa, buf_type)


def free_tcm(b: Builder, buf: Value) -> None:
    b.emit(f"memref.dealloc {buf.ssa} : {buf.type}")


def proton_mark(b: Builder, name: str, is_start: bool) -> None:
    action = "start" if is_start else "end"
    b.emit(f'proton.record {action} "{name}"')


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

def store_vec(b: Builder, ptr: Value, idx: Value, vec: Value) -> None:
    dst = _ranked(b, ptr)
    b.emit(f"vector.transfer_write {vec.ssa}, {dst.ssa}[{idx.ssa}]"
           f" {{in_bounds = [true]}} : {vec.type}, {dst.type}")


def store_scalar(b: Builder, ptr: Value, idx: Value, val: Value) -> None:
    dst = _ranked(b, ptr)
    b.emit(f"memref.store {val.ssa}, {dst.ssa}[{idx.ssa}] : {dst.type}")


def store_2d(b: Builder, ptr: Value, rows: int, cols: int, vec: Value) -> None:
    edtype = vec.type.vec_elem
    space = ptr.type.memref_space
    src = _ranked(b, ptr)
    mtype = _strided_2d_type(rows, cols, edtype, space)
    view = b.define("view2d", mtype,
                    f"memref.reinterpret_cast {src.ssa} to "
                    f"offset: [0], sizes: [{rows}, {cols}], strides: [{cols}, 1]"
                    f" : {src.type} to {mtype}")
    c0 = b.cint(0)
    b.emit(f"vector.transfer_write {vec.ssa}, {view.ssa}[{c0.ssa}, {c0.ssa}]"
           f" {{in_bounds = [true, true]}} : {vec.type}, {mtype}")


def store_2d_at(b: Builder, ptr: Value, off: Value, rows: int, cols: int,
                vec: Value) -> None:
    edtype = vec.type.vec_elem
    space = ptr.type.memref_space
    src = _ranked(b, ptr)
    rows_c = b.cint(rows)
    boff = b.define("boff", Type.index(), f"arith.muli {off.ssa}, {rows_c.ssa} : index")
    sp = f", {space}" if space else ""
    mtype = Type(f"memref<{rows}x{cols}x{edtype}, strided<[{cols}, 1], offset: ?>{sp}>")
    view = b.define("view2d", mtype,
                    f"memref.reinterpret_cast {src.ssa} to "
                    f"offset: [{boff.ssa}], sizes: [{rows}, {cols}], strides: [{cols}, 1]"
                    f" : {src.type} to {mtype}")
    c0 = b.cint(0)
    b.emit(f"vector.transfer_write {vec.ssa}, {view.ssa}[{c0.ssa}, {c0.ssa}]"
           f" {{in_bounds = [true, true]}} : {vec.type}, {mtype}")
