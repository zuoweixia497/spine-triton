from triton.language import core as tl
from triton._C.libtriton import ir

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

