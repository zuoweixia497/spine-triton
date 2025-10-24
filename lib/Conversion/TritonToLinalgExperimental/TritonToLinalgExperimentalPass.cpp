//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcileLlvmPtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ScfbufferStandardized.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Conversion/AddTargetDescription/AddTargetDescription.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ConvertScanOp.h"
#include "triton-shared/Conversion/XsmtToLinalg/XsmtToLinalg.h"
#include "mlir/Dialect/DLTI/DLTI.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "triton-shared/Dialect/xsmt/IR/XSMTDialect.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

class TritonToLinalgExperimentalPass
    : public TritonToLinalgExperimentalBase<TritonToLinalgExperimentalPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect,
                bufferization::BufferizationDialect, memref::MemRefDialect,
                ttx::TritonTilingExtDialect, tts::TritonStructuredDialect,
                tptr::TPtrDialect, ptr::PtrDialect, DLTIDialect, LLVM::LLVMDialect,
                vector::VectorDialect, xsmt::XSMTDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    PassManager pm(&getContext(), moduleOp.getOperationName());

    pm.addPass(createTritonToStructuredPass(
        enableMakeGatherScatterTensorPtr));

    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    pm.addPass(createTritonToUnstructuredPass());
    pm.addPass(createTritonArithToLinalgPass(/*tensorPtrToLinalg=*/true));
    pm.addPass(createScfbufferStandardizedPass());

    pm.addPass(createStructuredToMemrefPass());
    pm.addPass(createUnstructuredToMemrefPass());
    pm.addPass(createTritonPtrToMemrefPass());
    pm.addPass(createTritonToPtrPass());
    pm.addPass(createAddTargetDescriptionPass());
    pm.addPass(createConvertScanOpPass());
    // Now that remove-dead-values fully works with linalg ops, clean up the IR
    // again, particularly unused loop iter-args that were created
    // during triton-to-structured.
    pm.addPass(createRemoveDeadValuesPass());
    pm.addPass(createXsmtToLinalgPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createReconcilePtrCastsPass());
    pm.addPass(createReconcileLlvmPtrCastsPass());

    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (enableCollapseShape) {
      // TODO Enable this
      // Canonicalizer pass will rewrite tensor.expand_shape(linalg.fill) to
      // linalg.fill(tensor.expand_shape) so we need to run it before
      // collapseShape pass
      // pm.addPass(createCollapseShapePass());
    }

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToLinalgExperimentalPass() {
  return std::make_unique<TritonToLinalgExperimentalPass>();
}
