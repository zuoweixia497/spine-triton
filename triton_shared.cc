#include "include/triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "ir.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ir;
using namespace mlir;
namespace xsmt = mlir::xsmt;

void init_triton_xsmt_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
      ->def("create_annotation",
            [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
               Attribute &attrVal) {
              auto annotationOp = self.create<xsmt::AnnotationOp>(ptr);
              annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                    attrVal);
            })
      .def("create_descriptor_load",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
              std::vector<int32_t> &shape,
              std::vector<int32_t> &micro_size) -> Value {
             return self.create<xsmt::DescriptorLoadOp>(base, offsets, shape,
                                                        micro_size);
           })
      .def("create_descriptor_load_view",
           [](TritonOpBuilder &self, Value &base, Value &view) -> Value {
             return self.create<xsmt::DescriptorLoadViewOp>(base, view);
           })
      .def("create_view",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
              std::vector<int32_t> &shape,
              std::vector<int32_t> &micro_size) -> Value {
             return self.create<xsmt::ViewOp>(base, offsets, shape, micro_size);
           })
      .def("create_alloc",
           [](TritonOpBuilder &self, std::vector<int32_t> &shape,
              std::vector<int32_t> &micro_size, Type &elementType) -> Value {
             return self.create<xsmt::AllocOp>(shape, micro_size, elementType);
           })
      .def("create_mmt4d",
           [](TritonOpBuilder &self, Value &a, Value &b, Value &c) -> Value {
             auto aType = cast<RankedTensorType>(a.getType());
             auto bType = cast<RankedTensorType>(b.getType());

             assert(aType.getRank() == 4 && "A must be 4D packed tensor");
             assert(bType.getRank() == 4 && "B must be 4D packed tensor");

             auto aShape = aType.getShape();
             auto bShape = bType.getShape();
             assert(aShape[1] == bShape[0] && "KB dimension must match");
             assert(aShape[3] == bShape[2] && "kb dimension must match");

             SmallVector<int64_t> outputShape = {
                 aShape[0] * aShape[2],
                 bShape[1] * bShape[3],
             };
             auto resultType =
                 RankedTensorType::get(outputShape, aType.getElementType());
             auto perm = std::vector<int>{1, 0, 3, 2};
             auto transb = self.create<mlir::triton::TransOp>(b, perm);
             return self.create<xsmt::MMT4DOp>(resultType, a, transb, c);
           });
}

void init_triton_spine_triton(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::xsmt::XSMTDialect, tensor::TensorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_xsmt_ir(m.def_submodule("xsmt_ir"));
}
