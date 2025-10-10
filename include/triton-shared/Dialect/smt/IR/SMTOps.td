
#ifndef SMT_OPS
#define SMT_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/IR/EnumAttr.td"
// include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
// include "mlir/Interfaces/ControlFlowInterfaces.td" // RegionBranchOpInterface
// include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
// include "mlir/Interfaces/InferTypeOpInterface.td"  // SameOperandsAndResultType
include "triton-shared/Dialect/smt/IR/SMTDialect.td"


class SMT_Op<string mnemonic, list<Trait> traits = []> :
    Op<SMT_Dialect, mnemonic, traits>;

def AnnotationOp : SMT_Op<"annotation", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Annotate a tensor with key-value attribute pairs";
  let description = [{
    `smt.annotation` operation can be used to annotate a tensor with
    key-value attribute pairs.

    Example:
    ```mlir
    smt.annotation %target {key : val}
    ```
  }];
  let arguments = (ins TT_Tensor:$src);
  let assemblyFormat = [{
    $src attr-dict `:` type($src)
  }];
}

#endif
