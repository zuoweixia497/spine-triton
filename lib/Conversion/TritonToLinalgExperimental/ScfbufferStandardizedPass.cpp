#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ScfbufferStandardized.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

struct ScfbufferStandardizedPass : public ScfbufferStandardizedBase<ScfbufferStandardizedPass> {
public:
  void runOnOperation() override {
    Operation *module = getOperation();
    module->walk([&](scf::ForOp forOp) {
      processForOp(forOp);
    });
  }
private:
  void processForOp(scf::ForOp forOp) {
    Block *body = forOp.getBody();
    auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

    SmallVector<Value> iterArgs(forOp.getRegionIterArgs().begin(),
                                forOp.getRegionIterArgs().end());

    SmallVector<Value> yieldValues(yieldOp.getOperands().begin(),
                                   yieldOp.getOperands().end());

    if (iterArgs.size() != yieldValues.size())
      return;

    SmallVector<linalg::GenericOp> genericOps;
    SmallVector<int> resultIndices;

    for (auto [idx, yieldVal] : llvm::enumerate(yieldValues)) {
      Operation *defOp = yieldVal.getDefiningOp();
      if (!defOp) continue;

      if (auto genericOp = dyn_cast<linalg::GenericOp>(defOp)) {
        auto results = genericOp->getResults();
        for (int resIdx = 0; resIdx < results.size(); ++resIdx) {
          if (results[resIdx] == yieldVal) {
            genericOps.push_back(genericOp);
            resultIndices.push_back(resIdx);
            break;
          }
        }
      }
    }

    for (auto [genericOp, resIdx] : llvm::zip(genericOps, resultIndices)) {
      auto yieldVal = genericOp.getResult(resIdx);
      int argIdx = -1;
      for (auto [i, val] : llvm::enumerate(yieldValues)) {
        if (val == yieldVal) {
          argIdx = i;
          break;
        }
      }
      if (argIdx == -1) continue;

      Value iterArg = iterArgs[argIdx];
      auto outputs = genericOp.getOutputs();

      if (outputs.size() > resIdx && outputs[resIdx] != iterArg) {
        SmallVector<Value> newOutputs = outputs;
        newOutputs[resIdx] = iterArg;
        OpBuilder builder(genericOp);
        genericOp.getOutputsMutable().assign(newOutputs);
      }
    }

    for (auto [genericOp, resIdx] : llvm::zip(genericOps, resultIndices)) {
      auto result = genericOp.getResult(resIdx);

      int argIdx = -1;
      for (auto [i, val] : llvm::enumerate(yieldValues)) {
        if (val == result) {
          argIdx = i;
          break;
        }
      }
      if (argIdx == -1) continue;

      Value iterArg = iterArgs[argIdx];
      Value newResult = genericOp.getResult(resIdx);

      SmallVector<OpOperand *> usesToReplace;
      for (OpOperand &use : iterArg.getUses()) {
        Operation *user = use.getOwner();
        if (user != genericOp && user->getBlock() == body &&
            genericOp->isBeforeInBlock(user)) {
          usesToReplace.push_back(&use);
        }
      }

      for (OpOperand *use : usesToReplace) {
        if (auto userGeneric = dyn_cast<linalg::GenericOp>(use->getOwner())) {
          if (use->getOperandNumber() >= userGeneric.getInputs().size())
            continue;
        }
        use->set(newResult);
      }
    }
  }

};
} // namespace


std::unique_ptr<OperationPass<ModuleOp>> triton::createScfbufferStandardizedPass() {
  return std::make_unique<ScfbufferStandardizedPass>();
}

