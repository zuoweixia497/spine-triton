#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/AddTargetDescription/AddTargetDescription.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "add-target-description"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/AddTargetDescription/Passes.h.inc"

namespace {

class AddTargetDescriptionPass
    : public AddTargetDescriptionBase<AddTargetDescriptionPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                DLTIDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    auto cacheSizesAttr =
        moduleOp->getAttrOfType<mlir::DenseIntElementsAttr>("tt.cache_sizes");
    moduleOp->removeAttr("tt.cache_sizes");

    int32_t l1 = cacheSizesAttr.getValues<int32_t>()[0];
    int32_t l2 = cacheSizesAttr.getValues<int32_t>()[1];
    int32_t l3 = cacheSizesAttr.getValues<int32_t>()[2];

    SmallVector<mlir::DataLayoutEntryInterface> dlEntries;

    auto u64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    dlEntries.push_back(mlir::DataLayoutEntryAttr::get(
        mlir::StringAttr::get(ctx, "TCM_cache_size_in_bytes"),
        mlir::IntegerAttr::get(u64Type, 131072)));
    dlEntries.push_back(mlir::DataLayoutEntryAttr::get(
        mlir::StringAttr::get(ctx, "L1_cache_size_in_bytes"),
        mlir::IntegerAttr::get(u64Type, l1)));
    dlEntries.push_back(mlir::DataLayoutEntryAttr::get(
        mlir::StringAttr::get(ctx, "L2_cache_size_in_bytes"),
        mlir::IntegerAttr::get(u64Type, l2)));
    dlEntries.push_back(mlir::DataLayoutEntryAttr::get(
        mlir::StringAttr::get(ctx, "L3_cache_size_in_bytes"),
        mlir::IntegerAttr::get(u64Type, l3)));
    dlEntries.push_back(mlir::DataLayoutEntryAttr::get(
        mlir::StringAttr::get(ctx, "num_threads"),
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 4)));
    dlEntries.push_back(mlir::DataLayoutEntryAttr::get(
        mlir::StringAttr::get(ctx, "max_vector_width"),
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 256)));

    auto deviceSpec = mlir::TargetDeviceSpecAttr::get(ctx, dlEntries);

    SmallVector<mlir::DataLayoutEntryInterface> systemEntries;
    systemEntries.push_back(mlir::DataLayoutEntryAttr::get(
        mlir::StringAttr::get(ctx, "CPU"), deviceSpec));

    moduleOp->setAttr(mlir::DLTIDialect::kTargetSystemDescAttrName,
                      mlir::TargetSystemSpecAttr::get(ctx, systemEntries));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createAddTargetDescriptionPass() {
  return std::make_unique<AddTargetDescriptionPass>();
}