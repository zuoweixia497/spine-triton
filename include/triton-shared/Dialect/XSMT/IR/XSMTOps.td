
#ifndef XSMT_OPS
#define XSMT_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton-shared/Dialect/XSMT/IR/XSMTDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"


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


def ViewOp : XSMT_Op<"view", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Create a local view of a tensor with new shape and micro tile size";
  let description = [{
    Creates a local view of a tensor with the specified shape and micro tile size.
    This operation reshapes a tensor in shared memory for tensor core operations.
  }];

  let arguments = (ins
    TT_Tensor:$base,
    Variadic<I32>:$offsets,
    DenseI32ArrayAttr:$shape,
    DenseI32ArrayAttr:$micro_size
  );

  let results = (outs
    TT_Tensor:$result
  );

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

  let hasVerifier = 1;
}


def ViewPtrOp : XSMT_Op<"viewptr", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Create a local view of a tensor with new shape and micro tile size";
  let description = [{
    Creates a local view of a tensor with the specified shape and micro tile size.
    This operation reshapes a tensor in shared memory for tensor core operations.
  }];

  let arguments = (ins
    TT_TensorPtr:$base,
    Variadic<I32>:$offsets,
    DenseI32ArrayAttr:$shape,
    DenseI32ArrayAttr:$micro_size
  );

  let results = (outs
    TT_TensorPtr:$result
  );

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

  let hasVerifier = 1;
}

def DescriptorLoadViewOp : XSMT_Op<"descriptor_load_view", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "fuse DescriptorLoadOp and ViewOp";
  let description = [{
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

def AllocOp : XSMT_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a tensor in specified memory storage";
  let description = [{
    Allocates memory for a tensor with given shape in specified storage (e.g., l2, shared).
  }];

  let arguments = (ins
    DenseI32ArrayAttr:$shape,
    StrAttr:$storage
  );

  let results = (outs TT_TensorPtr:$result);

  let assemblyFormat = [{
    `[` $shape `]` `storage` `(` $storage `)` attr-dict `:` type($result)
  }];
}


def MMT4DOp : XSMT_Op<"mmt4d", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Matrix multiplication with 4D packed tensors";
  let description = [{
    The `xsmt.mmt4d` operation performs matrix multiplication on two 4D packed tensors.
    The operation expects the two inputs to be packed with the same inner_tile on the K dimension.

    If `c` is provided, performs `result = a @ b + c` (accumulation).
    If `c` is not provided, performs `result = a @ b` (no accumulation).

    Example:
    ```mlir
    // With accumulator
    %result = xsmt.mmt4d %a, %b, %c :
        (tensor<4x8x8x16xf32>, tensor<8x8x8x16xf32>, tensor<4x8x8x8xf32>) -> tensor<4x8x8x8xf32>

    // Without accumulator
    %result += xsmt.mmt4d %a, %b :
        (tensor<4x8x8x16xf32>, tensor<8x8x8x16xf32>) -> tensor<4x8x8x8xf32>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$a,
    TT_Tensor:$b,
    Optional<TT_Tensor>:$c
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $a `,` $b (`,` $c^)? attr-dict `:` type($a) `,` type($b) (`,` type($c)^)? `->` type($result)
  }];
}

def BindOp : XSMT_Op<"bind", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Bind a source value to a destination";
  let description = [{
    The `xsmt.bind` operation binds the source value to the destination.
    It returns the source value with the type of the destination.

    This operation is typically used for variable binding or value assignment
    in the XSMT dialect. The result type always matches the destination type.

    Example:
    ```mlir
    %bound = xsmt.bind %dest, %src : (tensor<f32>, tensor<f32>) -> tensor<f32>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$dest,
    TT_Tensor:$src
  );

  let results = (outs
    TT_Tensor:$result
  );

  let assemblyFormat = [{
    $dest `,` $src attr-dict `:` `(` type($dest) `,` type($src) `)` `->` type($result)
  }];
}

def XSMT_MBarrierCreateOp : XSMT_Op<"mbarrier_create"> {
  let summary = "Create a software barrier for thread synchronization.";

  let description = [{
    Creates a software barrier handle with specified configuration:
    - `flag`: Barrier mode (0=normal, 1=async, 2=with fence)
    - `atc`: Initial arrival thread count
    - `tc`: Total threads that must arrive before release
    - `exp`: Expected version value after barrier release

    Returns an opaque barrier handle for use with arrive/wait operations.
  }];

  let arguments = (ins
    I16:$flag,
    I16:$atc,
    I16:$tc,
    I16:$exp
  );

  let results = (outs TT_Ptr:$barrier);

  let assemblyFormat = [{
    $flag `,` $atc `,` $tc `,` $exp attr-dict `:` type($barrier)
  }];

}

def XSMT_BarrierArriveOp : XSMT_Op<"barrier_arrive"> {
  let summary = "Signal thread arrival at software barrier.";

  let description = [{
    Signals that the calling thread has arrived at the barrier.
    Decrements the pending arrival count. When count reaches zero,
    waiting threads are released.
  }];

  let arguments = (ins TT_Ptr:$barrier);

  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";

}

def XSMT_BarrierWaitOp : XSMT_Op<"barrier_wait"> {
  let summary = "Wait for barrier to reach expected version.";

  let description = [{
    Blocks the calling thread until the barrier reaches the expected
    version value, indicating all required threads have arrived.

    - `flag`: Wait mode (0=normal, 1=async, 2=with fence)
    - `exp`: Expected version to wait for
  }];

  let arguments = (ins
    TT_Ptr:$barrier,
    I16:$flag,
    I16:$exp
  );

  let assemblyFormat = [{
    $barrier `,` $flag `,` $exp attr-dict `:` type($barrier)
  }];

}

def GetThreadOp : XSMT_Op<"get_num_of_thread", [Pure, MemoryEffects<[MemWrite]>]> {

  let summary = "Get the current thread";

  let description = [{
    Returns the thread ID of the current context.
    Example:
      %0 = xsmt.get_num_of_thread
  }];

  let arguments = (ins);

  let results = (outs I32:$result);

  let assemblyFormat = "attr-dict";
}


def XSMT_GlobalMBarrierInitOp : XSMT_Op<"global_mbarrier_init", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create a global barrier.";

  let description = [{

  }];

  let arguments = (ins I16:$id);

  let results = (outs I64:$barrier);

  let assemblyFormat = [{
    $id attr-dict `:` type($barrier)
  }];

}

def XSMT_BarrierSetEepectOp : XSMT_Op<"barrier_set_expect"> {
  let summary = "Wait for barrier to reach expected version.";

  let description = [{
  }];

  let arguments = (ins
    I64:$barrier,
    I16:$exp
  );

  let assemblyFormat = [{
    $barrier `,` $exp attr-dict `:` type($barrier)
  }];

}

#endif
