# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT

from triton.language.core import (
    _unwrap_if_constexpr,
    builtin,
    constexpr,
    range,
)
from . import semantic as smt_semantic
from triton.language import core as tl


def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


@builtin
def compile_hint(ptr, hint_name, hint_val=None, _semantic=None):
    hint_name = _constexpr_to_value(hint_name)
    assert isinstance(hint_name, str), f"hint name: {hint_name} is not string"
    hint_val = _unwrap_if_constexpr(hint_val) if hint_val else hint_val
    smt_semantic.compile_hint(ptr, hint_name, hint_val, _semantic)


class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.

    This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
    :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param bind_sub_block: Tells the compiler if multiple vector cores participate in the loop.
        This is used in the mixed cube-vector kernel on 910B. The number of vector cores is determined by the number of
        iteration in this loop. Currently on 910B, max 2 vector cores could be used.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, bind_sub_block: bool = True):
        super().__init__(arg1, arg2, step, num_stages, loop_unroll_factor)
        self.bind_sub_block = bind_sub_block


@builtin
def descriptor_load(base, offsets_or_view, shape=None, micro_size=None, _semantic=None):
    """Descriptor-based block load operation with two forms:

    Form 1: descriptor_load(base, offsets, shape, micro_size)
    Form 2: descriptor_load(base, view)

    :param base: the base tensor pointer to load from
    :param offsets_or_view: either variadic offset values (e.g., [s * SUB_BLK_M, 0])
                           or a view tensor
    :param shape: shape of the block to load (e.g., [SUB_BLK_M, BLOCK_SIZE_N])
    :param micro_size: micro tile size for tensor cores (e.g., [8, 8])

    Examples
    *******
    .. code-block:: python

        # Form 1
        a_packed = smt.descriptor_load(a_block_ptr, [s * SUB_BLK_M, 0], [SUB_BLK_M, BLOCK_SIZE_N], [8, 8])
        # Form 2
        b_packed = smt.descriptor_load(b_block_ptr, b_packed_shared_view)
    """
    # Handle Form 2: descriptor_load(base, view)
    if shape is None and micro_size is None:
        return smt_semantic.descriptor_load_view(base, offsets_or_view, _semantic)

    # Handle Form 1: descriptor_load(base, offsets, shape, micro_size)
    return smt_semantic.descriptor_load(base, offsets_or_view, shape, micro_size, _semantic)


@builtin
def view(base, offsets, shape, micro_size,  _semantic=None):
    """Create a local view of a tensor with new shape and micro tile size.

    :param base: the base tensor to create view from
    :param shape: new shape for the view (e.g., [SUB_BLK_M, BLOCK_SIZE_K])
    :param micro_size: micro tile size for tensor cores (e.g., [16, 8])

    Example
    *******
    .. code-block:: python

        accumulator = tl.local_view(accumulator, [SUB_BLK_M, BLOCK_SIZE_K], [16, 8])
    """
    return smt_semantic.view(base, offsets, shape, micro_size, _semantic)


@builtin
def alloc(shape, micro_size, dtype=tl.float32, _semantic=None):
    """Allocate a tensor in shared memory with specified shape and micro tile size.

    :param shape: shape of the tensor to allocate (e.g., [BLOCK_SIZE_N, BLOCK_SIZE_K])
    :param micro_size: micro tile size for tensor cores (e.g., [16, 8])
    :param dtype: data type of the tensor elements (default: float32)

    Example
    *******
    .. code-block:: python

        b_packed_shared = tl.alloc([BLOCK_SIZE_N, BLOCK_SIZE_K], [16, 8])
    """
    return smt_semantic.alloc(shape, micro_size, dtype, _semantic)


@builtin
def dot(a_packed, b_packed, out_unpacked, _semantic=None):
    """
    Args:
        a_packed: packed A matrix, shape [MB, KB, mb, kb] (from MxK)
        b_packed: packed B matrix, shape [NB, KB, kb, nb] (from NxK)
    Returns:
        Matrix multiplication result, shape [MB, NB, mb, nb]

    """

    # assert len(a_packed.shape) == 4, f"A must be 4D packed, got {a_packed.shape}D"
    # assert len(b_packed.shape) == 4, f"B must be 4D packed, got {b_packed.shape}D"

    # # Check K dimension matches
    # assert a_packed.shape[1] == b_packed.shape[1], f"KB dim mismatch: A{a_packed.shape[1]} vs B{b_packed.shape[1]}"
    # assert a_packed.shape[3] == b_packed.shape[3], f"kb dim mismatch: A{a_packed.shape[3]} vs B{b_packed.shape[3]}"

    return smt_semantic.mmt4d(a_packed, b_packed, out_unpacked, _semantic)
