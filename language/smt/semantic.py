from triton.language import core as tl
from typing import List
from triton.language import semantic as tl_semantic

def compile_hint(ptr: tl.tensor, hint_name: str, hint_val, _semantic=None):
    if not hint_val:
        hint_val = _semantic.builder.get_unit_attr()
    elif isinstance(hint_val, bool):
        hint_val = _semantic.builder.get_bool_attr(hint_val)
    elif isinstance(hint_val, int):
        hint_val = _semantic.builder.get_int32_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    _semantic.builder.create_annotation(ptr.handle, hint_name, hint_val)


def descriptor_load(base: tl.tensor, offsets, shape, micro_size, _semantic=None) -> tl.tensor:
    semantic_instance = tl_semantic.TritonSemantic(_semantic.builder)
    offsets = semantic_instance._convert_to_ir_values(offsets)
    shape = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in shape]
    micro_size = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in micro_size]

    assert all(isinstance(elem, int) and -2**31 <= elem < 2**31 for elem in shape), \
        "Expected a list of constant integers (`int32_t` range) in `shape`"
    assert all(isinstance(elem, int) and -2**31 <= elem < 2**31 for elem in micro_size), \
        "Expected a list of constant integers (`int32_t` range) in `micro_size`"

    assert len(shape) == len(micro_size), \
        "Expected shape and micro_size to have the same length"

    handle = _semantic.builder.create_descriptor_load(base.handle, offsets, shape, micro_size)

    result_shape = [
        shape[0] // micro_size[0],
        shape[1] // micro_size[1],
        micro_size[0],
        micro_size[1]
    ]

    element_ty = base.type.element_ty.element_ty
    return tl.tensor(handle, tl.block_type(element_ty, result_shape))


def descriptor_load_view(base: tl.tensor, view: tl.tensor, _semantic=None) -> tl.tensor:
    """Form 2: descriptor_load(base, view)"""
    handle = _semantic.builder.create_descriptor_load_view(base.handle, view.handle)
    result_tensor = tl.tensor(handle, view.type)
    return result_tensor

def view(base: tl.tensor, offsets, shape, micro_size, _semantic=None) -> tl.tensor:
    semantic_instance = tl_semantic.TritonSemantic(_semantic.builder)
    offsets = semantic_instance._convert_to_ir_values(offsets)

    shape = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in shape]
    micro_size = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in micro_size]

    assert all(isinstance(elem, int) and -2**31 <= elem < 2**31 for elem in shape), \
        "Expected a list of constant integers (`int32_t` range) in `shape`"
    assert all(isinstance(elem, int) and -2**31 <= elem < 2**31 for elem in micro_size), \
        "Expected a list of constant integers (`int32_t` range) in `micro_size`"
    assert len(shape) == len(micro_size), \
        "Shape and micro_size must have the same length"

    for i, (dim, micro) in enumerate(zip(shape, micro_size)):
        assert dim % micro == 0, \
            f"Dimension {i} of shape ({dim}) must be divisible by micro_size ({micro})"

    handle = _semantic.builder.create_view(base.handle, offsets, shape, micro_size)

    result_shape = [
        shape[0] // micro_size[0],
        shape[1] // micro_size[1],
        micro_size[0],
        micro_size[1]
    ]

    element_ty = base.type.element_ty
    return tl.tensor(handle, tl.block_type(element_ty, result_shape))

def alloc(shape, micro_size, dtype, _semantic=None) -> tl.tensor:
    dtype = dtype.to_ir(_semantic.builder)
    shape = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in shape]
    micro_size = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in micro_size]

    assert all(isinstance(elem, int) and -2**31 <= elem < 2**31 for elem in shape), \
        "Expected a list of constant integers (`int32_t` range) in `shape`"
    assert all(isinstance(elem, int) and -2**31 <= elem < 2**31 for elem in micro_size), \
        "Expected a list of constant integers (`int32_t` range) in `micro_size`"
    assert len(shape) == len(micro_size), \
        "Shape and micro_size must have the same length"

    for i, (dim, micro) in enumerate(zip(shape, micro_size)):
        assert dim % micro == 0, \
            f"Dimension {i} of shape ({dim}) must be divisible by micro_size ({micro})"

    handle = _semantic.builder.create_alloc(shape, micro_size, dtype)

    result_shape = [
        shape[0] // micro_size[0],
        shape[1] // micro_size[1],
        micro_size[0],
        micro_size[1]
    ]

    return tl.tensor(handle, tl.block_type(dtype, result_shape))

def mmt4d(a_packed: tl.tensor, b_packed: tl.tensor, out_unpacked: tl.tensor, _semantic=None):
    assert len(a_packed.shape) == 4, f"A must be 4D packed, got {a_packed.shape}D"
    assert len(b_packed.shape) == 4, f"B must be 4D packed, got {b_packed.shape}D"
    assert a_packed.shape[1] == b_packed.shape[0], f"KB dim mismatch: A{a_packed.shape[1]} vs B{b_packed.shape[0]}"
    assert a_packed.shape[3] == b_packed.shape[2], f"nb dim mismatch: B{b_packed.shape[3]} vs out{b_packed.shape[2]}"

    mb = a_packed.shape[0]
    nb = b_packed.shape[1]
    mb_micro_sizes = a_packed.shape[2]
    nb_micro_sizes = b_packed.shape[3]
    output_shape = [mb * mb_micro_sizes, nb * nb_micro_sizes]
    ret_type = tl.block_type(a_packed.type.scalar, output_shape)
    out = _semantic.builder.create_mmt4d(a_packed.handle, b_packed.handle, out_unpacked.handle)
    return tl.tensor(out, ret_type)
