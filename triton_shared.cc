#include <pybind11/pybind11.h>
#include "ir.h" // TritonOpBuilder
#include "mlir/Pass/PassManager.h"
#include "include/triton-shared/Dialect/smt/IR/SMTDialect.h"

namespace py = pybind11;
using namespace ir;
using namespace mlir;
namespace smt = mlir::smt;

void init_triton_smt_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls->def("create_annotation", [](TritonOpBuilder &self, Value &ptr,
                                            const std::string &attrKey,
                                            Attribute &attrVal) {
    auto annotationOp = self.create<smt::AnnotationOp>(ptr);
    annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey), attrVal);
  });
}

void init_triton_spine_triton(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::smt::SMTDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_smt_ir(m.def_submodule("smt_ir"));
}
