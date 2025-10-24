
#ifndef XSMT_OPS
#define XSMT_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "triton-shared/Dialect/xsmt/IR/XSMTDialect.td"


class XSMT_Op<string mnemonic, list<Trait> traits = []> :
    Op<XSMT_Dialect, mnemonic, traits>;

def AnnotationOp : XSMT_Op<"annotation", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Annotate a tensor with key-value attribute pairs";
  let description = [{
    `xsmt.annotation` operation can be used to annotate a tensor with
    key-value attribute pairs.

    Example:
    ```mlir
    xsmt.annotation %target {key : val}
    ```
  }];
  let arguments = (ins TT_Tensor:$src);
  let assemblyFormat = [{
    $src attr-dict `:` type($src)
  }];
}


def DescriptorLoadOp : XSMT_Op<"descriptor_load", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Descriptor-based block load operation";
  let description = [{
    Loads a block of data from memory using tensor core descriptors.
    This operation uses descriptor-based addressing for efficient tensor core access patterns.
  }];

  let arguments = (ins
    TT_Ptr:$base,
    Variadic<I64>:$offsets,
    DenseI32ArrayAttr:$shape,
    DenseI32ArrayAttr:$micro_size
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $base `,` `[` $offsets `]` `,` `[` $shape `]` `,` `[` $micro_size `]` attr-dict `:`
    type($base) `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "ArrayRef<int32_t>":$shape,
      "ArrayRef<int32_t>":$micro_size
    )>
  ];
}

def DescriptorLoadViewOp : XSMT_Op<"descriptor_load_view", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Descriptor-based block load operation with view";
  let description = [{
    Loads a block of data from memory using tensor core descriptors and a view.
    This form uses a pre-defined view tensor to specify the access pattern.
  }];

  let arguments = (ins
    TT_Ptr:$base,
    TT_Tensor:$view
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $base `,` $view attr-dict `:` type($base) `,` type($view) `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$base,
      "Value":$view
    )>
  ];
}

def ViewOp : XSMT_Op<"view", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Create a local view of a tensor with new shape and micro tile size";
  let description = [{
    Creates a local view of a tensor with the specified shape and micro tile size.
    This operation reshapes a tensor in shared memory for tensor core operations.
  }];

  let arguments = (ins
    TT_Tensor:$base,
    Variadic<I64>:$offsets,
    DenseI32ArrayAttr:$shape,
    DenseI32ArrayAttr:$micro_size
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $base `,` `[` $offsets `]` `,` `[` $shape `]` `,` `[` $micro_size `]` attr-dict `:`
    type($base) `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$base,
      "ValueRange":$offsets,
      "ArrayRef<int32_t>":$shape,
      "ArrayRef<int32_t>":$micro_size
    )>
  ];
}

def AllocOp : XSMT_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a tensor in shared memory with specified shape and micro tile size";
  let description = [{
    Allocates a tensor in shared memory with the specified shape and micro tile size.
    This operation is commonly used for allocating intermediate results in matrix multiplication
    and other tensor operations that require specific memory layouts for tensor cores.
  }];

  let arguments = (ins
    DenseI32ArrayAttr:$shape,
    DenseI32ArrayAttr:$micro_size
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    `[` $shape `]` `,` `[` $micro_size `]` attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "ArrayRef<int32_t>":$shape,
      "ArrayRef<int32_t>":$micro_size,
      "Type":$elementType
    )>
  ];
}


def MMT4DOp : XSMT_Op<"mmt4d", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Matrix multiplication with 4D packed tensors";
  let description = [{
    The `xsmt.mmt4d` operation performs matrix multiplication on two 4D packed tensors.
    The operation expects the two inputs to be packed with the same inner_tile on the K dimension.

    Example:
    ```mlir
    %result = xsmt.mmt4d %a, %b :
        (tensor<4x8x8x16xf32>, tensor<8x8x8x16xf32>) -> tensor<4x8x8x8xf32>
    ```
  }];

  let arguments = (ins TT_Tensor:$a, TT_Tensor:$b, TT_Tensor:$c);

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $a `,` $b`,` $c attr-dict `:` type($a) `,` type($b) `,` type($c) `->` type($result)
  }];

  // let hasVerifier = 1;
}

#endif
