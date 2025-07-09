//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
// Throughout the conversion process, we convert !tt.ptr -> {!ptr.ptr or
// memref<*>}. This process leaves around unrealized_conversion_cast ops between
// these types. We want to remove these unrealized casts and use the proper
// conversion ops in the PtrDialect: to_memref or from_memref. To do this, we
// use a pattern that simplifies the chain of conversions by removing
// intermediate conversion cast ops. At the end, we are left with just pointer
// to memref or vice versa. We then convert the unrealized cast to to_memref or
// from_memref accordingly.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcileLlvmPtrCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcileLlvmPtrCasts.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {


struct PromoteMemrefToPtrArg : public OpRewritePattern<func::FuncOp> {
  PromoteMemrefToPtrArg(MLIRContext *context)
      : OpRewritePattern<func::FuncOp>(context) {}

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    Block &entryBlock = funcOp.getBody().front();
    MLIRContext *ctx = funcOp->getContext();
    SmallVector<UnrealizedConversionCastOp> castOps;

    for (BlockArgument arg : entryBlock.getArguments()) {
      for (OpOperand &use : arg.getUses()) {
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(use.getOwner())) {
          if (castOp.getNumResults() == 1 &&
            isLLVMPtrType(castOp.getResult(0).getType())) {
            castOps.push_back(castOp);
          }
        }
      }
    }

    if (castOps.empty())
      return failure();

    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type> newArgTypes(oldType.getInputs());

    for (auto &castOp : castOps) {
      newArgTypes.push_back(castOp.getResult(0).getType());
    }

    auto newFuncType = FunctionType::get(ctx, newArgTypes, oldType.getResults());

    Location loc = funcOp.getLoc();
    func::FuncOp newFunc = rewriter.create<func::FuncOp>(
        loc, funcOp.getName(), newFuncType);

    newFunc->setAttrs(funcOp->getAttrs());

    newFunc.setType(newFuncType);

    Block *newEntryBlock = rewriter.createBlock(&newFunc.getBody());
    for (Type type : newArgTypes) {
      newEntryBlock->addArgument(type, loc);
    }

    for (int i = 0, e = oldType.getNumInputs(); i < e; i++) {
      if (DictionaryAttr attrs = funcOp.getArgAttrDict(i)) {
        newFunc.setArgAttrs(i, attrs.getValue());
      }
    }
    IRMapping mapper;

    unsigned numOrigArgs = oldType.getNumInputs();
    for (unsigned i = 0; i < numOrigArgs; i++) {
      mapper.map(entryBlock.getArgument(i), newEntryBlock->getArgument(i));
    }

    for (unsigned i = 0; i < castOps.size(); i++) {
      mapper.map(castOps[i].getResult(0), newEntryBlock->getArgument(numOrigArgs + i));
    }

    for (Operation &op : entryBlock.getOperations()) {
      if (isa<UnrealizedConversionCastOp>(op)) {
        continue;
      }
      rewriter.clone(op, mapper);
    }

    rewriter.replaceOp(funcOp, newFunc->getResults());

    return success();
  }
private:
  bool isLLVMPtrType(Type type) const {
    if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type)) {
      return true;
    }
    return false;
  }
};


class ReconcileLlvmPtrCastsPass
    : public ReconcileLlvmPtrCastsBase<ReconcileLlvmPtrCastsPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns
        .add<PromoteMemrefToPtrArg>(
            &getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createReconcileLlvmPtrCastsPass() {
  return std::make_unique<ReconcileLlvmPtrCastsPass>();
}
