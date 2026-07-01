# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""SpineMLIRCodeGenerator — translates @spine_raw Python functions to Linalg MLIR.

AST visitor for the spine_raw eDSL subset. Resolves each AST node to an
ir.Value and dispatches builtin calls to typed constructors in ops.py; the
Builder (ir.py) owns SSA naming, constant pooling and indentation. This is the
typed-builder rewrite of the original f-string codegen — see
docs/spine_raw_v2_migration_plan.md. Output is byte-identical to the original
(tests/test_spine_raw_golden.py).

Supports: In/InOut-annotated params, scf.for with auto iter_args, index/vector
BinOps, and the spine_raw builtins (splat/load_vec/.../batch_macc/pack_2d_t...).
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable

from . import ops
from .ir import Builder, Type, Value
from .types import _TypedAnnotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_signature(fn: Callable) -> list[tuple[str, _TypedAnnotation]]:
    sig = inspect.signature(fn)
    result = []
    for pname, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            raise ValueError(
                f"Parameter '{pname}' of @spine_raw function '{fn.__name__}' "
                f"must have an In[...] or InOut[...] annotation."
            )
        if not isinstance(ann, _TypedAnnotation):
            raise ValueError(
                f"Parameter '{pname}' annotation must be In[...] or InOut[...], got {ann!r}"
            )
        result.append((pname, ann))
    return result


def _find_reassigned(body: list, outer_vars: set) -> set:
    """Variables in outer_vars that are assigned inside body (direct stmts only)."""
    found = set()
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id in outer_vars:
                    found.add(t.id)
    return found


def _eval_list_literal(node) -> list:
    if isinstance(node, ast.List):
        return [ast.literal_eval(e) for e in node.elts]
    return [ast.literal_eval(node)]


def _is_spine_raw_attr(node, attr: str, aliases: set | None = None) -> bool:
    """Check if node is <alias>.<attr> where alias is a spine_raw module import."""
    if not (isinstance(node, ast.Attribute) and node.attr == attr):
        return False
    if not isinstance(node.value, ast.Name):
        return False
    if aliases is not None:
        return node.value.id in aliases
    return True


_BIN_INT = {ast.Add: "addi", ast.Mult: "muli", ast.Sub: "subi"}
_BIN_FLOAT = {ast.Add: "addf", ast.Mult: "mulf", ast.Sub: "subf"}


# ---------------------------------------------------------------------------
# SpineMLIRCodeGenerator
# ---------------------------------------------------------------------------

class SpineMLIRCodeGenerator(ast.NodeVisitor):
    """Translate a @spine_raw Python function to a func.func MLIR string.

    Supported subset:
      - Function parameters with In[...] / InOut[...] annotations
      - spine_raw builtins (see ops.py) dispatched by name
      - for VAR in spine_raw.range(N): with automatic iter_arg detection
      - BinOp + / * / - on index or matching vector types
      - integer / float constants
    """

    def __init__(self):
        self._aliases: set[str] = {"spine_raw", "sr"}
        self._reset()

    def _reset(self):
        self.b = Builder()
        self._env: dict[str, Value] = {}
        self._all_iter_arg_names: set[str] = set()
        self._loop_iter_args: set[str] = set()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self, fn: Callable) -> str:
        """Return a bare func.func @name(...) { ... } MLIR string for fn."""
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not func_nodes:
            raise ValueError(f"No function definition found in {fn.__name__!r}")

        # Detect all aliases for the spine_raw module in the function's globals
        try:
            import spine_raw as _sr_mod
        except ModuleNotFoundError:
            try:
                from triton.language.extra import spine_raw as _sr_mod
            except (ModuleNotFoundError, ImportError):
                _sr_mod = None

        aliases: set[str] = set()
        if _sr_mod is not None:
            for k, v in (fn.__globals__ or {}).items():
                if v is _sr_mod:
                    aliases.add(k)
        self._aliases = aliases or {"spine_raw", "sr"}

        return self._gen_func(func_nodes[0], fn)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _bind(self, name: str, value: Value):
        self._env[name] = value

    def _get(self, name: str) -> Value:
        if name not in self._env:
            raise ValueError(f"Undefined variable: {name!r}")
        return self._env[name]

    # ------------------------------------------------------------------
    # Function-level generation
    # ------------------------------------------------------------------

    def _gen_func(self, node: ast.FunctionDef, fn: Callable) -> str:
        self._reset()

        params = _parse_signature(fn)
        fname = node.name

        for pname, ann in params:
            self.b.reserve(pname)
            self._env[pname] = Value(f"%{pname}", Type(ann.mlir_type))

        # Pre-scan: variables that will become iter_args in for loops.
        defined_so_far: set[str] = set(self._env.keys())
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        defined_so_far.add(t.id)
            elif isinstance(stmt, ast.For):
                self._all_iter_arg_names |= _find_reassigned(stmt.body, defined_so_far)

        sig_parts = [f"    %{pname} : {ann.mlir_type}" for pname, ann in params]
        header = f"func.func @{fname}(\n" + ",\n".join(sig_parts) + "\n) {"

        for stmt in node.body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # skip docstrings
            self._gen_stmt(stmt)

        self.b.emit("return")

        return f"{header}\n{self.b.render_body()}\n}}"

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _gen_stmt(self, node):
        if isinstance(node, ast.Assign):
            self._gen_assign(node)
        elif isinstance(node, ast.For):
            self._gen_for(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self._gen_call_stmt(node.value)
        elif isinstance(node, (ast.Return, ast.Pass)):
            pass
        else:
            raise NotImplementedError(f"Unsupported statement: {ast.dump(node)}")

    def _gen_assign(self, node: ast.Assign):
        assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        target = node.targets[0].id

        if target in self._loop_iter_args:
            hint = f"{target}_upd"
        elif target in self._all_iter_arg_names:
            hint = f"{target}_init"
        else:
            hint = target

        value = self._gen_expr(node.value, hint=hint)
        self._bind(target, value)

    def _gen_for(self, node: ast.For):
        assert isinstance(node.target, ast.Name), "for target must be a simple name"
        loop_var = node.target.id

        assert _is_spine_raw_attr(node.iter.func, "range", self._aliases), \
            "for loop iter must be spine_raw.range(N)"
        assert len(node.iter.args) == 1, "spine_raw.range takes one argument"
        ub = self._gen_expr(node.iter.args[0])

        c0 = self.b.cint(0)
        c1 = self.b.cint(1)

        outer_vars = set(self._env.keys())
        iter_args = sorted(_find_reassigned(node.body, outer_vars))
        ia_data = [(v, self._get(v)) for v in iter_args]  # (name, init Value)

        result_ssas = {v: self.b.new(v) for v in iter_args}
        loop_ssa = self.b.new(loop_var)
        self._bind(loop_var, Value(loop_ssa, Type.index()))

        if ia_data:
            res_part = ", ".join(result_ssas[v] for v in iter_args)
            ia_part = ", ".join(f"%{v}_in = {init.ssa}" for v, init in ia_data)
            types_part = ", ".join(str(init.type) for _, init in ia_data)
            self.b.emit(
                f"{res_part} = scf.for {loop_ssa} = {c0.ssa} to {ub.ssa} step {c1.ssa}"
                f" iter_args({ia_part}) -> ({types_part}) {{"
            )
        else:
            self.b.emit(f"scf.for {loop_ssa} = {c0.ssa} to {ub.ssa} step {c1.ssa} {{")

        # Inside the loop, iter_arg vars resolve to the %<v>_in block args.
        for v, init in ia_data:
            in_name = f"{v}_in"
            self.b.reserve(in_name)
            self._bind(v, Value(f"%{in_name}", init.type))

        prev_loop_iter_args = self._loop_iter_args
        self._loop_iter_args = set(iter_args)

        self.b.indent(2)
        for stmt in node.body:
            self._gen_stmt(stmt)

        if ia_data:
            yields = [self._get(v) for v in iter_args]
            self.b.emit(
                f"scf.yield {', '.join(y.ssa for y in yields)}"
                f" : {', '.join(str(y.type) for y in yields)}"
            )

        self.b.indent(-2)
        self.b.emit("}")

        self._loop_iter_args = prev_loop_iter_args

        # After the loop, iter_arg vars resolve to the scf.for results.
        for v, init in ia_data:
            self._bind(v, Value(result_ssas[v], init.type))

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def _gen_expr(self, node, hint: str = "") -> Value:
        if isinstance(node, ast.Name):
            return self._get(node.id)
        if isinstance(node, ast.Constant):
            return self._gen_literal(node)
        if isinstance(node, ast.BinOp):
            return self._gen_binop(node, hint)
        if isinstance(node, ast.Call):
            return self._gen_call_expr(node, hint)
        raise NotImplementedError(f"Unsupported expr: {ast.dump(node)}")

    def _gen_literal(self, node: ast.Constant) -> Value:
        v = node.value
        if isinstance(v, bool):
            raise NotImplementedError(f"Unsupported literal: {v!r}")
        if isinstance(v, int):
            return self.b.cint(v)
        if isinstance(v, float):
            return self.b.cfloat(v)
        raise NotImplementedError(f"Unsupported literal: {v!r}")

    def _gen_binop(self, node: ast.BinOp, hint: str) -> Value:
        lval = self._gen_expr(node.left)
        rval = self._gen_expr(node.right)
        op = type(node.op)
        result = self.b.new(hint or "t")

        if lval.type.is_index and rval.type.is_index:
            opname = _BIN_INT.get(op)
            if opname is None:
                raise NotImplementedError(f"BinOp {op.__name__} not supported for index")
            self.b.emit(f"{result} = arith.{opname} {lval.ssa}, {rval.ssa} : index")
            return Value(result, Type.index())

        if lval.type == rval.type and lval.type.is_vector:
            opname = _BIN_FLOAT.get(op)
            if opname is None:
                raise NotImplementedError(f"BinOp {op.__name__} not supported for {lval.type}")
            self.b.emit(f"{result} = arith.{opname} {lval.ssa}, {rval.ssa} : {lval.type}")
            return Value(result, lval.type)

        raise NotImplementedError(
            f"BinOp between {lval.type!r} and {rval.type!r} not supported")

    # ------------------------------------------------------------------
    # Call dispatch — argument extraction lives here, emission in ops.py
    # ------------------------------------------------------------------

    @staticmethod
    def _kw(node):
        return {kw.arg: kw.value for kw in node.keywords}

    def _opt(self, kwargs, args, idx, name):
        """Optional Value arg from positional idx or keyword name; None if absent."""
        n = args[idx] if idx < len(args) else kwargs.get(name)
        return self._gen_expr(n) if n is not None else None

    def _m_arg(self, node):
        """Stride/dim arg: Python int when a static literal, else an ir.Value."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        return self._gen_expr(node)

    def _gen_call_expr(self, node: ast.Call, hint: str) -> Value:
        a, kw = node.args, self._kw(node)

        def pick(i, name):
            return a[i] if i < len(a) else kw[name]

        def attr(name):
            return _is_spine_raw_attr(node.func, name, self._aliases)

        if attr("splat"):
            val = self._gen_expr(a[0] if a else kw["val"])
            shape = _eval_list_literal(kw["shape"])
            assert len(shape) == 1, "spine_raw.splat only supports 1D shape"
            return ops.splat(self.b, val, shape[0], hint)
        if attr("load_vec"):
            ptr = self._gen_expr(a[0])
            idx = self._gen_expr(a[1])
            n = ast.literal_eval(a[2]) if len(a) > 2 else ast.literal_eval(kw["N"])
            dt_node = kw.get("dtype") or (a[3] if len(a) > 3 else None)
            dtype = ast.literal_eval(dt_node) if dt_node else "f32"
            ib = ast.literal_eval(kw["in_bounds"]) if kw.get("in_bounds") else True
            return ops.load_vec(self.b, ptr, idx, n, dtype, ib, hint)
        if attr("extf"):
            v = self._gen_expr(a[0])
            dt = ast.literal_eval(a[1]) if len(a) > 1 else "f32"
            return ops.extf(self.b, v, dt, hint)
        if attr("fma"):
            return ops.fma(self.b, self._gen_expr(a[0]), self._gen_expr(a[1]),
                           self._gen_expr(a[2]), hint)
        if attr("reduce_add"):
            return ops.reduce_add(self.b, self._gen_expr(a[0]), hint)
        if attr("matmul"):
            lhs = self._gen_expr(pick(0, "lhs"))
            rhs = self._gen_expr(pick(1, "rhs"))
            acc = self._gen_expr(pick(2, "acc"))
            m = ast.literal_eval(pick(3, "m"))
            n = ast.literal_eval(pick(4, "n"))
            k = ast.literal_eval(pick(5, "k"))
            return ops.matmul(self.b, lhs, rhs, acc, m, n, k, hint)
        if attr("load_tile"):
            ptr = self._gen_expr(pick(0, "ptr"))
            rb = self._gen_expr(pick(1, "row_base"))
            cb = self._gen_expr(pick(2, "col_base"))
            row_stride = ast.literal_eval(pick(3, "row_stride"))
            m = ast.literal_eval(pick(4, "M"))
            k = ast.literal_eval(pick(5, "K"))
            dt_node = a[6] if len(a) > 6 else kw.get("dtype")
            dtype = ast.literal_eval(dt_node) if dt_node else "f16"
            return ops.load_tile(self.b, ptr, rb, cb, row_stride, m, k, dtype, hint)
        if attr("pad_vec"):
            return ops.pad_vec(self.b, self._gen_expr(a[0]), ast.literal_eval(a[1]), hint)
        if attr("extract_elem"):
            v = self._gen_expr(a[0])
            idx_node = a[1]
            if isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, int):
                idx = idx_node.value
            else:
                idx = self._gen_expr(idx_node)
            return ops.extract_elem(self.b, v, idx, hint)
        if attr("batch_macc"):
            return ops.batch_macc(self.b, self._gen_expr(a[0]), self._gen_expr(a[1]),
                                  self._gen_expr(a[2]), hint)
        if attr("view_2d"):
            ptr = self._gen_expr(a[0])
            rows = ast.literal_eval(a[1])
            cols = ast.literal_eval(a[2])
            dtype = ast.literal_eval(a[3]) if len(a) > 3 else "f16"
            off = self._opt(kw, a, 4, "off")
            return ops.view_2d(self.b, ptr, rows, cols, dtype, off, hint)
        if attr("load_2d"):
            ptr = self._gen_expr(a[0])
            rows = ast.literal_eval(a[1])
            cols = ast.literal_eval(a[2])
            dtype = ast.literal_eval(a[3]) if len(a) > 3 else "f16"
            return ops.load_2d(self.b, ptr, rows, cols, dtype, hint)
        if attr("load_2d_at"):
            ptr = self._gen_expr(a[0])
            off = self._gen_expr(a[1])
            rows = ast.literal_eval(a[2])
            cols = ast.literal_eval(a[3])
            dtype = ast.literal_eval(a[4]) if len(a) > 4 else "f16"
            return ops.load_2d_at(self.b, ptr, off, rows, cols, dtype, hint)
        if attr("load_2d_t"):
            ptr = self._gen_expr(a[0])
            rb = self._gen_expr(a[1])
            k = ast.literal_eval(a[2])
            nb = ast.literal_eval(a[3])
            m = self._m_arg(a[4])
            dtype = ast.literal_eval(a[5]) if len(a) > 5 else "f16"
            col_off = self._opt(kw, a, 6, "col_off")
            return ops.load_2d_t(self.b, ptr, rb, k, nb, m, dtype, col_off, hint)
        if attr("pack_2d_t"):
            ptr = self._gen_expr(a[0])
            rb = self._gen_expr(a[1])
            k = ast.literal_eval(a[2])
            nb = ast.literal_eval(a[3])
            m = self._m_arg(a[4])
            dtype = ast.literal_eval(a[5]) if len(a) > 5 else "f16"
            col_off = self._opt(kw, a, 6, "col_off")
            return ops.pack_2d_t(self.b, ptr, rb, k, nb, m, dtype, col_off, hint)
        if attr("alloc_tcm_2d"):
            k = ast.literal_eval(a[0])
            nb = ast.literal_eval(a[1])
            dtype = ast.literal_eval(a[2]) if len(a) > 2 else "f16"
            return ops.alloc_tcm_2d(self.b, k, nb, dtype, hint)
        if attr("splat_2d"):
            val = self._gen_expr(a[0])
            rows = ast.literal_eval(a[1])
            cols = ast.literal_eval(a[2])
            dtype = ast.literal_eval(a[3]) if len(a) > 3 else "f32"
            return ops.splat_2d(self.b, val, rows, cols, dtype, hint)
        raise NotImplementedError(f"Unsupported call: {ast.dump(node.func)}")

    def _gen_call_stmt(self, node: ast.Call):
        a, kw = node.args, self._kw(node)

        def attr(name):
            return _is_spine_raw_attr(node.func, name, self._aliases)

        if attr("store_vec"):
            return ops.store_vec(self.b, self._gen_expr(a[0]), self._gen_expr(a[1]),
                                 self._gen_expr(a[2]))
        if attr("store_scalar"):
            return ops.store_scalar(self.b, self._gen_expr(a[0]), self._gen_expr(a[1]),
                                    self._gen_expr(a[2]))
        if attr("store_2d"):
            ptr = self._gen_expr(a[0])
            rows = ast.literal_eval(a[1])
            cols = ast.literal_eval(a[2])
            vec = self._gen_expr(a[3])
            return ops.store_2d(self.b, ptr, rows, cols, vec)
        if attr("store_2d_at"):
            ptr = self._gen_expr(a[0])
            off = self._gen_expr(a[1])
            rows = ast.literal_eval(a[2])
            cols = ast.literal_eval(a[3])
            vec = self._gen_expr(a[4])
            return ops.store_2d_at(self.b, ptr, off, rows, cols, vec)
        if attr("pack_2d_t_into"):
            buf = self._gen_expr(a[0])
            ptr = self._gen_expr(a[1])
            rb = self._gen_expr(a[2])
            k = ast.literal_eval(a[3])
            nb = ast.literal_eval(a[4])
            m = self._m_arg(a[5])
            dtype = ast.literal_eval(a[6]) if len(a) > 6 else "f16"
            col_off = self._opt(kw, a, 7, "col_off")
            return ops.pack_2d_t_into(self.b, buf, ptr, rb, k, nb, m, dtype, col_off)
        if attr("free_tcm"):
            return ops.free_tcm(self.b, self._gen_expr(a[0]))
        if attr("proton_mark"):
            name = ast.literal_eval(a[0])
            is_start = len(a) >= 2 and ast.literal_eval(a[1])
            return ops.proton_mark(self.b, name, is_start)
        raise NotImplementedError(f"Unsupported call statement: {ast.dump(node.func)}")
