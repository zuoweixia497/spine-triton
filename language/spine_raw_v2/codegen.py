# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""spine_raw_v2 codegen — FlagTree-style AST executor over mlir.ir Python bindings.

The user writes raw MLIR using mlir.dialects ops directly. The codegen walks the
Python AST and EXECUTES it in an MLIR context — no string generation.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable, Dict, Optional, Sequence

from mlir import ir
from mlir.dialects import func as func_d


class SpineIRCodeGenerator(ast.NodeVisitor):
    """AST executor for spine_raw_v2 kernels.

    Design (mirrors FlagTree's MLIRCodeGenerator):
      - user writes MLIR Python bindings calls directly: arith.constant(...),
        scf.ForOp(...), vector.TransferReadOp(...)
      - visit_Call just evaluates the Python expression — the MLIR Python API
        creates ir.Operation objects directly at the current insertion point.
      - no string generation. no dispatch tables.
    """

    def __init__(self, fn: Callable, context: Optional[ir.Context] = None):
        super().__init__()
        self._fn = fn
        self._globals: Dict[str, Any] = dict(fn.__globals__ or {})
        self._context = context or ir.Context()
        self._context.allow_unregistered_dialects = True
        self._module: Optional[ir.Module] = None
        self._func_op: Optional[func_d.FuncOp] = None

    # ── public API ────────────────────────────────────────────────

    def generate(self) -> ir.Module:
        """Walk the AST and return an ir.Module."""
        src = textwrap.dedent(inspect.getsource(self._fn))
        tree = ast.parse(src)
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_node = node
                break
        if func_node is None:
            raise ValueError(f"No function definition found in {self._fn.__name__!r}")

        with self._context, ir.Location.unknown():
            self._module = ir.Module.create()
            self._visit_func(func_node)
        return self._module

    # ── AST visitors ──────────────────────────────────────────────

    def _visit_func(self, node: ast.FunctionDef):
        from .types import In, InOut

        params = []
        for a in node.args.args:
            ann = a.annotation
            mtype = ir.IndexType.get()  # default to index for scalars
            is_memref = False
            if ann is not None:
                try:
                    ann_obj = ast.literal_eval(ann)
                    if hasattr(ann_obj, 'mlir_type'):
                        mtype = ann_obj.mlir_type
                    if isinstance(ann_obj, InOut):
                        is_memref = True
                    elif isinstance(ann_obj, In):
                        is_memref = True
                except (ValueError, TypeError):
                    pass
            params.append((a.arg, mtype))

        arg_types = [p[1] for p in params]
        ftype = ir.FunctionType.get(arg_types, [])
        self._func_op = func_d.FuncOp(node.name, ftype)
        self._func_op.visibility = "private"
        self._module.body.append(self._func_op)

        entry = self._func_op.add_entry_block()
        with ir.InsertionPoint(entry):
            for i, (pname, _) in enumerate(params):
                self._globals[pname] = entry.arguments[i]

            for stmt in node.body:
                if isinstance(stmt, ast.Pass):
                    continue
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    continue
                self._visit_stmt(stmt)

            if not entry.operations or not _is_terminator(entry.operations[-1]):
                func_d.ReturnOp([])

    def _visit_stmt(self, node):
        if isinstance(node, ast.Assign):
            self._visit_assign(node)
        elif isinstance(node, ast.For):
            self._visit_for(node)
        elif isinstance(node, ast.Expr):
            self.visit(node.value)
        elif isinstance(node, ast.If):
            self._visit_if(node)
        else:
            self.visit(node)

    def _visit_assign(self, node: ast.Assign):
        val = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._globals[target.id] = val
            elif isinstance(target, ast.Tuple):
                for t, v in zip(target.elts, val):
                    if isinstance(t, ast.Name):
                        self._globals[t.id] = v

    def _visit_for(self, node: ast.For):
        from mlir.dialects import scf

        # Expect: for kb in sr_mod.range(nk):
        if isinstance(node.iter, ast.Call):
            iter_fn = self.visit(node.iter.func)
            iter_args = [self.visit(a) for a in node.iter.args]
            upper_bound = iter_fn(*iter_args)
        else:
            upper_bound = self.visit(node.iter)

        c0 = ir.IntegerAttr.get(ir.IndexType.get(), 0)
        c1 = ir.IntegerAttr.get(ir.IndexType.get(), 1)
        zero_op = arith.ConstantOp(ir.IndexType.get(), c0)  # noqa: F821
        one_op = arith.ConstantOp(ir.IndexType.get(), c1)  # noqa: F821

        # determine iter_args from assignments in loop body
        iter_arg_names = _find_loop_iter_args(node, self._globals)
        init_vals = [self._globals[n] for n in iter_arg_names]

        loop = scf.ForOp(zero_op, upper_bound, one_op, init_vals)
        with ir.InsertionPoint(loop.body):
            self._globals[node.target.id] = loop.induction_variable
            for i, n in enumerate(iter_arg_names):
                self._globals[n] = loop.inner_iter_args[i]

            for stmt in node.body:
                self._visit_stmt(stmt)

        for i, n in enumerate(iter_arg_names):
            self._globals[n] = loop.results[i]

    def _visit_if(self, node: ast.If):
        cond = self.visit(node.test)
        from mlir.dialects import scf
        if_op = scf.IfOp(cond, [])
        with ir.InsertionPoint(if_op.then_block):
            for stmt in node.body:
                self._visit_stmt(stmt)
        if node.orelse:
            with ir.InsertionPoint(if_op.else_block):
                for stmt in node.orelse:
                    self._visit_stmt(stmt)

    # ── expression evaluation ─────────────────────────────────────

    def visit_Call(self, node: ast.Call):
        fn = self.visit(node.func)
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return fn(*args, **kwargs)

    def visit_Name(self, node: ast.Name):
        if node.id in self._globals:
            return self._globals[node.id]
        raise NameError(f"undefined: {node.id}")

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_Attribute(self, node: ast.Attribute):
        obj = self.visit(node.value)
        return getattr(obj, node.attr)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Mult):
            from mlir.dialects import arith
            return arith.MulIOp(left, right).result
        if isinstance(node.op, ast.Add):
            from mlir.dialects import arith
            return arith.AddIOp(left, right).result
        if isinstance(node.op, ast.Sub):
            from mlir.dialects import arith
            return arith.SubIOp(left, right).result
        raise NotImplementedError(f"unsupported binary op: {node.op}")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            from mlir.dialects import arith
            zero = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0))
            return arith.SubIOp(zero, self.visit(node.operand)).result
        raise NotImplementedError(f"unsupported unary op: {node.op}")

    def visit_Compare(self, node: ast.Compare):
        from mlir.dialects import arith
        left = self.visit(node.left)
        [right] = [self.visit(c) for c in node.comparators]
        if isinstance(node.ops[0], ast.Lt):
            return arith.CmpIOp(arith.CmpIPredicate.slt, left, right).result
        if isinstance(node.ops[0], ast.LtE):
            return arith.CmpIOp(arith.CmpIPredicate.sle, left, right).result
        if isinstance(node.ops[0], ast.Gt):
            return arith.CmpIOp(arith.CmpIPredicate.sgt, left, right).result
        if isinstance(node.ops[0], ast.GtE):
            return arith.CmpIOp(arith.CmpIPredicate.sge, left, right).result
        if isinstance(node.ops[0], ast.Eq):
            return arith.CmpIOp(arith.CmpIPredicate.eq, left, right).result
        raise NotImplementedError(f"unsupported compare op: {node.ops[0]}")


# ── helpers ────────────────────────────────────────────────────

def _find_loop_iter_args(for_node: ast.For, globals: dict) -> list[str]:
    """Find variable names that are assigned in the loop body but not defined there."""
    defined_in_body = set()
    for stmt in for_node.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    defined_in_body.add(t.id)

    # A variable used in the body before assignment → iter_arg candidate
    # Simple heuristic: any variable in globals that gets assigned in body
    return [n for n in defined_in_body if n in globals]


def _is_terminator(op) -> bool:
    return op.operation.name.endswith(".return") or op.operation.name.endswith(".yield")


# need to import arith for the constant ops used in _visit_for
from mlir.dialects import arith  # noqa: E402
