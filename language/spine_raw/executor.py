# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""Executor codegen for the direct-op DSL (dsl.py) — FlagTree-style.

Unlike SpineMLIRCodeGenerator (which dispatches `sr.<name>` markers through a
big if/elif), this visitor *executes* the kernel body: `visit_Call` resolves the
function from the kernel's globals and calls it, so `alloc_tcm_2d(32, 64, "f16")`
runs the real dsl.alloc_tcm_2d against the implicit Builder (ir.CTX). Control
flow that can't be Python-executed (the dynamic-trip-count loop, the func
signature) is still interpreted: `for kb in srange(n):` becomes scf.for with
auto-detected iter_args.

Because the executor pushes each assignment target as the result-name hint
(ir.CTX), a kernel written in the DSL produces byte-identical MLIR to the
original marker codegen — see tests/test_spine_raw_dsl.py.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable

from .codegen import _find_reassigned, _parse_signature
from .ir import CTX, Builder, Type, Value


class SpineDSLExecutor(ast.NodeVisitor):
    """Run a direct-op DSL kernel body, emitting MLIR into a Builder."""

    def __init__(self, fn: Callable):
        self._fn = fn
        self._globals = dict(fn.__globals__ or {})
        self.b = Builder()
        self._env: dict[str, Value] = {}
        self._all_iter_arg_names: set[str] = set()
        self._loop_iter_args: set[str] = set()
        self._srange = self._globals.get("srange")

    # ------------------------------------------------------------------
    def generate(self) -> str:
        src = textwrap.dedent(inspect.getsource(self._fn))
        tree = ast.parse(src)
        fn_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        prev = CTX.builder, CTX.pending
        CTX.builder, CTX.pending = self.b, None
        try:
            return self._gen_func(fn_node)
        finally:
            CTX.builder, CTX.pending = prev

    # ------------------------------------------------------------------
    def _gen_func(self, node: ast.FunctionDef) -> str:
        params = _parse_signature(self._fn)
        for pname, ann in params:
            self.b.reserve(pname)
            self._env[pname] = Value(f"%{pname}", Type(ann.mlir_type))

        # Pre-scan: which vars become scf.for iter_args.
        defined_so_far = set(self._env.keys())
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        defined_so_far.add(t.id)
            elif isinstance(stmt, ast.For):
                self._all_iter_arg_names |= _find_reassigned(stmt.body, defined_so_far)

        sig = [f"    %{p} : {ann.mlir_type}" for p, ann in params]
        header = f"func.func @{node.name}(\n" + ",\n".join(sig) + "\n) {"

        for stmt in node.body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # docstring
            self._exec(stmt)

        self.b.emit("return")
        return f"{header}\n{self.b.render_body()}\n}}"

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------
    def _exec(self, node):
        if isinstance(node, ast.Assign):
            self._exec_assign(node)
        elif isinstance(node, ast.For):
            self._exec_for(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            CTX.pending = None
            self.visit(node.value)
        elif isinstance(node, (ast.Return, ast.Pass)):
            pass
        else:
            raise NotImplementedError(f"Unsupported statement: {ast.dump(node)}")

    def _exec_assign(self, node: ast.Assign):
        assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        target = node.targets[0].id
        if target in self._loop_iter_args:
            hint = f"{target}_upd"
        elif target in self._all_iter_arg_names:
            hint = f"{target}_init"
        else:
            hint = target
        CTX.pending = hint
        value = self.visit(node.value)
        CTX.pending = None
        self._env[target] = value

    def _exec_for(self, node: ast.For):
        assert isinstance(node.target, ast.Name), "for target must be a simple name"
        loop_var = node.target.id
        assert (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and self._globals.get(node.iter.func.id) is self._srange), \
            "for loop iter must be srange(N)"
        assert len(node.iter.args) == 1, "srange takes one argument"
        ub = self.visit(node.iter.args[0])

        c0 = self.b.cint(0)
        c1 = self.b.cint(1)

        iter_args = sorted(_find_reassigned(node.body, set(self._env.keys())))
        ia_data = [(v, self._env[v]) for v in iter_args]
        result_ssas = {v: self.b.new(v) for v in iter_args}
        loop_ssa = self.b.new(loop_var)
        self._env[loop_var] = Value(loop_ssa, Type.index())

        if ia_data:
            res = ", ".join(result_ssas[v] for v in iter_args)
            ia = ", ".join(f"%{v}_in = {init.ssa}" for v, init in ia_data)
            tys = ", ".join(str(init.type) for _, init in ia_data)
            self.b.emit(f"{res} = scf.for {loop_ssa} = {c0.ssa} to {ub.ssa} step {c1.ssa}"
                        f" iter_args({ia}) -> ({tys}) {{")
        else:
            self.b.emit(f"scf.for {loop_ssa} = {c0.ssa} to {ub.ssa} step {c1.ssa} {{")

        for v, init in ia_data:
            in_name = f"{v}_in"
            self.b.reserve(in_name)
            self._env[v] = Value(f"%{in_name}", init.type)

        prev = self._loop_iter_args
        self._loop_iter_args = set(iter_args)
        self.b.indent(2)
        for stmt in node.body:
            self._exec(stmt)
        if ia_data:
            ys = [self._env[v] for v in iter_args]
            self.b.emit(f"scf.yield {', '.join(y.ssa for y in ys)}"
                        f" : {', '.join(str(y.type) for y in ys)}")
        self.b.indent(-2)
        self.b.emit("}")
        self._loop_iter_args = prev

        for v, init in ia_data:
            self._env[v] = Value(result_ssas[v], init.type)

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------
    def visit_Call(self, node: ast.Call):
        func = self.visit(node.func)
        saved = CTX.pending
        CTX.pending = None  # args must not consume the result's name hint
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        CTX.pending = saved
        return func(*args, **kwargs)

    def visit_Name(self, node: ast.Name):
        if node.id in self._env:
            return self._env[node.id]
        if node.id in self._globals:
            return self._globals[node.id]
        raise NameError(f"undefined name in spine_raw kernel: {node.id!r}")

    def visit_Attribute(self, node: ast.Attribute):
        return getattr(self.visit(node.value), node.attr)

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = type(node.op)
        if op is ast.Add:
            return left + right
        if op is ast.Mult:
            return left * right
        if op is ast.Sub:
            return left - right
        raise NotImplementedError(f"Unsupported binop: {op.__name__}")

    def visit_List(self, node: ast.List):
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node: ast.Tuple):
        return tuple(self.visit(e) for e in node.elts)
