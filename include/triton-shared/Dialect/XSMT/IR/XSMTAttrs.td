//===- XSMTAttrs.td - XSMT dialect attrs -------------------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef XSMT_ATTRS
#define XSMT_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "triton-shared/Dialect/XSMT/IR/XSMTDialect.td"

class XSMT_Attr<string name, string _mnemonic, list<Trait> traits = []>
    : AttrDef<XSMT_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def XSMT_MemorySpaceAttr
    : XSMT_Attr<"MemorySpace", "memory_space",
                 [DeclareAttrInterfaceMethods<MemorySpaceAttrInterface>]> {
  let summary = "Unified memory space attribute for spine-triton";
  let description = [{
    Represents a memory space in the spine-triton compilation pipeline.
    Used on memref types and ptr types to avoid MLIR's IntegerAttr(0)
    normalization issue (MLIR treats IntegerAttr(0) as default and drops it).

    Canonical scope names:
      global   — default address space (DDR / main memory)
      tcm      — tightly-coupled memory
      l2       — L2 cache / scratchpad
      fragment — register file / fragment

    Syntax: `#xsmt.memory_space<global>`
  }];

  let parameters = (ins StringRefParameter<"scope name">:$scope);
  let assemblyFormat = "`<` $scope `>`";
}

#endif // XSMT_ATTRS
