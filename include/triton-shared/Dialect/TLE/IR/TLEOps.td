#ifndef TLE_OPS
#define TLE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton-shared/Dialect/TLE/IR/TLEDialect.td"

class TLE_Op<string mnemonic, list<Trait> traits = []> :
    Op<TLE_Dialect, mnemonic, traits>;

// ============================================================================
// ToTensorOp / ToBufferOp
// ============================================================================
def TLE_ToTensorOp : TLE_Op<"to_tensor", [Pure]> {
  let summary = "Explicit buffer-to-tensor bridge";

  let description = [{
    Convert a pointer-to-block buffer value into tensor world. This is an
    explicit bridge op that lowers to bufferization.to_tensor.
  }];

  let arguments = (ins TT_TensorPtr:$buffer);
  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$buffer)>
  ];

  let assemblyFormat = [{
    $buffer attr-dict `:` qualified(type($buffer)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

def TLE_ToBufferOp : TLE_Op<"to_buffer", [MemoryEffects<[MemWrite]>]> {
  let summary = "Explicit tensor-to-buffer bridge";

  let description = [{
    Materialize a tensor-world value into an explicit pointer-to-block buffer
    destination. This bridge lowers to bufferization.materialize_in_destination.
  }];

  let arguments = (ins TT_Tensor:$tensor, TT_TensorPtr:$buffer);
  let results = (outs TT_TensorPtr:$result);

  let builders = [
    OpBuilder<(ins "Value":$tensor, "Value":$buffer)>
  ];

  let assemblyFormat = [{
    $tensor `in` $buffer attr-dict `:` qualified(type($tensor)) `,`
    qualified(type($buffer)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

// ============================================================================
// LocalPtrOp
// ============================================================================
def TLE_LocalPtrOp : TLE_Op<"local_ptr", [Pure]> {
  let summary = "Materialize local-buffer pointers from buffer + indices";

  let description = [{
    Generate a pointer (scalar or tensor of pointers) from a pointer-to-block
    buffer and optional index operands.

    Three index modes:
      - No indices: full-view pointer tensor covering the entire buffer shape.
      - All scalar indices: scalar pointer into the buffer.
      - All tensor indices (same shape): block pointer tensor of that shape.

    The result can be consumed by tl.load / tl.store directly.
  }];

  let arguments = (ins TT_TensorPtr:$buffer, Variadic<TT_IntLike>:$indices);
  let results = (outs TT_PtrLike:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// ============================================================================
// ExtractTileOp
// ============================================================================
def TLE_ExtractTileOp : TLE_Op<"extract_tile", [Pure]> {
  let summary = "Extract a tile from a tensor at a given tile index";

  let description = [{
    Extract a contiguous tile from a tensor based on a linearized tile index.
    The tile grid is computed from source shape / tile_shape (row-major order).

    Index can be:
      - Static (arith.constant): full compile-time validation
      - Dynamic (runtime value): only compile-time-checkable constraints verified

    Requirements:
      1. Source shape must be divisible by tile_shape in each dimension
      2. Result shape must equal tile_shape
      3. Element types must match
  }];

  let arguments = (ins TT_Tensor:$src, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$index,
        "ArrayRef<int64_t>":$tileShape)>
  ];

  let assemblyFormat = [{
    $src `[` $index `]` attr-dict `:` qualified(type($src)) `,`
    qualified(type($index)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

// ============================================================================
// InsertTileOp
// ============================================================================
def TLE_InsertTileOp : TLE_Op<"insert_tile", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Insert a tile into a tensor at a given tile index";

  let description = [{
    Insert a tile into a source tensor at the specified linearized tile index.
    Returns a new tensor with the tile values written at the target position.
    This is a functional (non-in-place) operation.

    The tile grid is computed from source shape / tile shape (row-major order).
  }];

  let arguments = (ins TT_Tensor:$src, TT_Tensor:$tile, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src `[` $index `]` `=` $tile attr-dict `:` qualified(type($src)) `,`
    qualified(type($index)) `,` qualified(type($tile)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

#endif
