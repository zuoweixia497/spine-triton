from triton.language.core import (
    _unwrap_if_constexpr,
    builtin,
    constexpr,
)
from . import semantic as dl_semantic

def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v

@builtin
def compile_hint(ptr, hint_name, hint_val=None, _semantic=None):
    hint_name = _constexpr_to_value(hint_name)
    assert isinstance(hint_name, str), f"hint name: {hint_name} is not string"
    hint_val = _unwrap_if_constexpr(hint_val) if hint_val else hint_val
    dl_semantic.compile_hint(ptr, hint_name, hint_val, _semantic)