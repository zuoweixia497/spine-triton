//===- MemorySpaceUtils.h - Unified memory space helpers --------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Provides a single source of truth for memory space representation across
// all Triton-to-memref conversion passes.
//
// Design: ALL memory spaces use xsmt::MemorySpaceAttr — a custom attribute
// that implements ptr::MemorySpaceAttrInterface.  Unlike IntegerAttr(0),
// MLIR does NOT normalize custom attrs to null, so the memory space is
// preserved on memref types and matches ptr::PtrType's memory space.
//
// Helpers:
//   getDefaultBridgeMemorySpace() — returns #xsmt.memory_space<global> for
//       the Triton ptr ↔ memref bridge (used by TritonToPtrPass, etc.).
//   scopeToMemorySpace()         — returns #xsmt.memory_space<scope> for
//       the xsmt.alloc / xsmt.alloc_copies / BufferType lowering path.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_UTILS_MEMORYSPACEUTILS_H
#define TRITON_SHARED_UTILS_MEMORYSPACEUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTAttrs.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton {

/// Return the canonical "bridge" memory space used when converting Triton
/// pointer types to memref types.  Returns #xsmt.memory_space<global>.
///
/// Every conversion pass that deals with tt.ptr ↔ memref should call this.
/// The returned xsmt::MemorySpaceAttr implements ptr::MemorySpaceAttrInterface,
/// so it is compatible with ptr::PtrType and all ptr dialect operations.
///
/// Note: Ensures the XSMT dialect is loaded in the context.
inline Attribute getDefaultBridgeMemorySpace(MLIRContext *ctx) {
  ctx->getOrLoadDialect<xsmt::XSMTDialect>();
  return xsmt::MemorySpaceAttr::get(ctx, "global");
}

/// Map a user-facing scope name to a memref memory-space attribute.
///
/// Canonical scope names:
///   "" / "global"  -> #xsmt.memory_space<global>
///   "tcm"          -> #xsmt.memory_space<tcm>
///   "l2"           -> #xsmt.memory_space<l2>
///   "fragment"     -> #xsmt.memory_space<fragment>
///
/// Unknown scopes fall back to #xsmt.memory_space<global>.
inline Attribute scopeToMemorySpace(llvm::StringRef scope, MLIRContext *ctx) {
  ctx->getOrLoadDialect<xsmt::XSMTDialect>();
  if (scope.empty() || scope == "global")
    return xsmt::MemorySpaceAttr::get(ctx, "global");
  if (scope == "tcm" || scope == "l2" || scope == "fragment")
    return xsmt::MemorySpaceAttr::get(ctx, scope);
  // Unknown scope — default to global.
  return xsmt::MemorySpaceAttr::get(ctx, "global");
}

} // namespace mlir::triton

#endif // TRITON_SHARED_UTILS_MEMORYSPACEUTILS_H
