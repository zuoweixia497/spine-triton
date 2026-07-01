# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""@spine_raw decorator and SpineLinalgJITFunction.

SpineLinalgJITFunction wraps a Python function annotated with In/InOut,
triggers SpineMLIRCodeGenerator on first call to make_linalg(), and
caches the resulting MLIR string.
"""
from __future__ import annotations

import inspect
from typing import Callable

from .codegen import SpineMLIRCodeGenerator


class SpineLinalgJITFunction:
    """Wrapper around a @spine_raw function that compiles Python → Linalg MLIR.

    Attributes:
        _fn         : original Python function
        _mlir_text  : cached bare func.func string (None until make_linalg() called)
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self._mlir_text: str | None = None
        # Tell Triton's JIT not to track this as a mutable global (same as
        # FlagTree's MLIRJITFunction.__triton_builtin__)
        self.__triton_builtin__ = True

    @property
    def __name__(self) -> str:
        return self._fn.__name__

    def make_linalg(self) -> str:
        """Trigger AST → MLIR compilation (lazy, cached).

        Returns a module-wrapped MLIR string:
            module {
              func.func @name(...) { ... }
            }
        """
        if self._mlir_text is None:
            gen = SpineMLIRCodeGenerator()
            func_text = gen.generate(self._fn)
            self._mlir_text = "module {{\n{}\n}}\n".format(func_text)
        return self._mlir_text

    def __repr__(self) -> str:
        return f"SpineLinalgJITFunction({self._fn.__name__!r})"


_REGISTRY: dict[str, type] = {
    "linalg": SpineLinalgJITFunction,
}


class SpineDSLJITFunction:
    """Wrapper for a direct-op DSL kernel (dsl.py + executor.py).

    Same external contract as SpineLinalgJITFunction (make_linalg() + __name__),
    so spine_raw.call() injects it through the unchanged create_tle_dsl_region
    path. The body is run by SpineDSLExecutor instead of the marker codegen.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self._mlir_text: str | None = None
        self.__triton_builtin__ = True

    @property
    def __name__(self) -> str:
        return self._fn.__name__

    def make_linalg(self) -> str:
        if self._mlir_text is None:
            from .executor import SpineDSLExecutor
            body = SpineDSLExecutor(self._fn).generate()
            self._mlir_text = "module {{\n{}\n}}\n".format(body)
        return self._mlir_text

    def __repr__(self) -> str:
        return f"SpineDSLJITFunction({self._fn.__name__!r})"


_REGISTRY["dsl"] = SpineDSLJITFunction


def spine_kernel(fn: Callable) -> SpineDSLJITFunction:
    """Decorator for a direct-op DSL kernel (the FlagTree-style writing surface).

    Equivalent to @spine_raw(name="dsl"). Usage:

        from spine_raw import spine_kernel, In, InOut
        from spine_raw.dsl import alloc_tcm_2d, batch_macc, srange, ...

        @spine_kernel
        def mv_macc_block(B: In[...], ..., C: InOut[...]):
            buf0 = alloc_tcm_2d(32, 64, "f16")
            ...
    """
    return SpineDSLJITFunction(fn)



def spine_raw(*, name: str = "linalg") -> Callable:
    """Decorator: mark a Python function as a raw Linalg MLIR kernel.

    Usage:
        @spine_raw(name="linalg")
        def mv_acc_raw_inner(A: In["memref<*xf16, #ptr.generic_space>"], ...):
            ...
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"spine_raw: unknown backend {name!r}. Available: {list(_REGISTRY)}"
        )
    cls = _REGISTRY[name]

    def decorator(fn: Callable) -> SpineLinalgJITFunction:
        return cls(fn)

    return decorator
