# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""Typed MLIR builder for spine_raw (no external `mlir` package required).

This replaces the original "f-string everywhere" codegen with a small typed
layer, mirroring FlagTree's `mlir.ir`-based approach but self-contained (the
spine-triton LLVM install ships no MLIR Python bindings — see
docs/spine_raw_v2_migration_plan.md):

    Type    — an MLIR type wrapped with structured accessors; constructed via
              parse()/vector()/memref helpers so a wrong shape/dtype is a
              Python error at build time, not an llc-stage crash.
    Value   — an SSA value: an MLIR name (e.g. "%lhs") plus its Type.
    Builder — owns SSA-name allocation, constant pooling, indentation and the
              line buffer. Op constructors live in ops.py and drive this.

The Builder's name/constant/indent bookkeeping is byte-for-byte compatible with
the original SpineMLIRCodeGenerator so the refactor is verifiable against the
captured goldens (tests/test_spine_raw_golden.py).
"""
import re


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

class Type:
    """An MLIR type, carried as canonical text plus structured accessors.

    Canonical text is kept verbatim so rendering is exact; accessors decode it
    for validation (is_memref / vec dims / element dtype).
    """

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    # -- rendering / identity ------------------------------------------------
    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Type({self.text!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Type) and other.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)

    # -- predicates ----------------------------------------------------------
    @property
    def is_index(self) -> bool:
        return self.text == "index"

    @property
    def is_vector(self) -> bool:
        return self.text.startswith("vector<")

    @property
    def is_memref(self) -> bool:
        return self.text.startswith("memref<")

    @property
    def is_unranked_memref(self) -> bool:
        return self.text.startswith("memref<*x")

    # -- vector decoding -----------------------------------------------------
    @property
    def vec_n(self) -> int:
        """First dimension of a vector<...> type."""
        m = re.match(r"vector<(\d+)x", self.text)
        if not m:
            raise ValueError(f"Cannot extract size from {self.text!r}")
        return int(m.group(1))

    @property
    def vec_rest(self) -> str:
        """Everything after the first 'x' of a vector type (matches the old
        _vec_elem helper: vector<32xf16> -> 'f16', vector<1x64xf32> ->
        '64xf32')."""
        m = re.match(r"vector<\d+x(.+)>", self.text)
        if not m:
            raise ValueError(f"Cannot extract elem type from {self.text!r}")
        return m.group(1)

    @property
    def vec_elem(self) -> str:
        """Scalar element dtype of a 1-D or 2-D vector (last x-token)."""
        m = re.match(r"vector<[\dx]+x([a-z0-9]+)>", self.text)
        if not m:
            raise ValueError(f"Cannot extract scalar elem from {self.text!r}")
        return m.group(1)

    # -- memref decoding -----------------------------------------------------
    @property
    def memref_space(self) -> str:
        """Address-space attribute of a memref type, or '' if none."""
        m = re.search(r",\s*(#[\w.]+)>", self.text)
        return m.group(1) if m else ""

    def to_ranked(self) -> "Type":
        """memref<*xT...> -> memref<?xT...> (unranked to 1-D ranked)."""
        if self.is_unranked_memref:
            return Type(self.text.replace("memref<*x", "memref<?x", 1))
        return self

    # -- constructors --------------------------------------------------------
    @staticmethod
    def parse(text: str) -> "Type":
        """Validate light well-formedness (balanced <>) and wrap as Type."""
        if text.count("<") != text.count(">"):
            raise ValueError(f"Malformed MLIR type (unbalanced <>): {text!r}")
        return Type(text)

    @staticmethod
    def index() -> "Type":
        return Type("index")

    @staticmethod
    def scalar(dtype: str) -> "Type":
        return Type(dtype)

    @staticmethod
    def vector(shape, dtype: str) -> "Type":
        dims = "x".join(str(d) for d in shape)
        return Type(f"vector<{dims}x{dtype}>")


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------

class Value:
    """An SSA value: MLIR name ('%foo') paired with its Type.

    Supports the index/vector arithmetic that a kernel body writes with plain
    Python operators (kb * 32, col + 64). Each operator emits an arith op on the
    current Builder (see CTX) — this is what lets the FlagTree-style executor
    run a kernel written as direct op calls without a marker dispatch table.
    """

    __slots__ = ("ssa", "type")

    def __init__(self, ssa: str, type: Type):
        self.ssa = ssa
        self.type = type

    def __repr__(self) -> str:
        return f"Value({self.ssa}, {self.type})"

    # -- arithmetic (emitted on CTX.builder) ---------------------------------
    def __add__(self, other):
        return _arith(self, other, "addi", "addf")

    def __radd__(self, other):
        return _arith(other, self, "addi", "addf")

    def __mul__(self, other):
        return _arith(self, other, "muli", "mulf")

    def __rmul__(self, other):
        return _arith(other, self, "muli", "mulf")

    def __sub__(self, other):
        return _arith(self, other, "subi", "subf")

    def __rsub__(self, other):
        return _arith(other, self, "subi", "subf")


# ---------------------------------------------------------------------------
# Execution context (implicit builder + pending SSA-name hint)
# ---------------------------------------------------------------------------

class _Context:
    """Holds the Builder a kernel body emits into, plus a one-shot SSA-name hint.

    The executor sets `builder` for the duration of a kernel and `pending` to the
    current assignment target name before evaluating its value expression; the
    first op/arith to emit consumes it via take_hint(), so `r1 = load_2d(...)`
    names its result %r1 — matching the original codegen's naming exactly.
    """

    def __init__(self):
        self.builder = None
        self.pending = None

    def set_hint(self, hint):
        self.pending = hint

    def take_hint(self) -> str:
        h = self.pending
        self.pending = None
        return h or ""


CTX = _Context()


def _as_value(x):
    """Coerce a Python int operand to an index constant Value."""
    if isinstance(x, Value):
        return x
    if isinstance(x, int):
        return CTX.builder.cint(x)
    raise TypeError(f"cannot use {x!r} as an MLIR operand")


def _arith(lhs, rhs, iop: str, fop: str) -> Value:
    b = CTX.builder
    lv = _as_value(lhs)
    rv = _as_value(rhs)
    hint = CTX.take_hint()
    result = b.new(hint or "t")
    if lv.type.is_index and rv.type.is_index:
        b.emit(f"{result} = arith.{iop} {lv.ssa}, {rv.ssa} : index")
        return Value(result, Type.index())
    if lv.type == rv.type and lv.type.is_vector:
        b.emit(f"{result} = arith.{fop} {lv.ssa}, {rv.ssa} : {lv.type}")
        return Value(result, lv.type)
    raise NotImplementedError(
        f"arith between {lv.type!r} and {rv.type!r} not supported")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class Builder:
    """Owns SSA naming, constant pooling, indentation and the line buffer.

    Emission semantics are byte-identical to the original codegen:
      * constants live in a preamble block at fixed 2-space indent;
      * SSA names dedup with a single shared counter suffix;
      * body lines carry the current indent.
    """

    def __init__(self):
        self._preamble = []          # constant defs, always indent 2
        self._lines = []             # body ops at current indent
        self._indent = 2
        self._counter = 0
        self._defined_ssas = set()
        self._const_ints = {}
        self._const_floats = {}

    # -- SSA allocation ------------------------------------------------------
    def new(self, hint: str) -> str:
        """Allocate a unique SSA name '%hint', deduping with a counter suffix."""
        if hint not in self._defined_ssas:
            self._defined_ssas.add(hint)
            return f"%{hint}"
        self._counter += 1
        candidate = f"{hint}_{self._counter}"
        while candidate in self._defined_ssas:
            self._counter += 1
            candidate = f"{hint}_{self._counter}"
        self._defined_ssas.add(candidate)
        return f"%{candidate}"

    def reserve(self, name: str) -> None:
        """Mark a bare name (e.g. a func param or scf block arg) as defined."""
        self._defined_ssas.add(name)

    # -- constant pooling ----------------------------------------------------
    def cint(self, n: int) -> Value:
        if n not in self._const_ints:
            name = f"c{abs(n)}" + ("" if n >= 0 else "_neg")
            ssa = self.new(name)
            self._const_ints[n] = ssa
            self._preamble.append(f"  {ssa} = arith.constant {n} : index")
        return Value(self._const_ints[n], Type.index())

    def cfloat(self, v: float, ftype: str = "f32") -> Value:
        key = (v, ftype)
        if key not in self._const_floats:
            if v == 0.0:
                hint = f"zero_{ftype}"
                lit = "0.000000e+00"
            else:
                hint = f"cf_{ftype}"
                lit = f"{v:e}"
            ssa = self.new(hint)
            self._const_floats[key] = ssa
            self._preamble.append(f"  {ssa} = arith.constant {lit} : {ftype}")
        return Value(self._const_floats[key], Type.scalar(ftype))

    # -- emission ------------------------------------------------------------
    def emit(self, line: str) -> None:
        self._lines.append(" " * self._indent + line)

    def indent(self, delta: int) -> None:
        self._indent += delta

    def define(self, hint: str, type: Type, rhs: str) -> Value:
        """Allocate a result name, emit '<name> = <rhs>', return the Value.

        The op constructors in ops.py format <rhs> (the right-hand side of an
        MLIR op). Centralising name allocation + emission here is what removes
        the scattered manual SSA bookkeeping of the old codegen.
        """
        name = self.new(hint)
        self.emit(f"{name} = {rhs}")
        return Value(name, type)

    # -- finalization --------------------------------------------------------
    def render_body(self) -> str:
        return "\n".join(self._preamble + self._lines)
