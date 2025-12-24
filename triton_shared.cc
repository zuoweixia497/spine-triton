#include "include/triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "include/triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"
#include "ir.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;
using namespace ir;
using namespace mlir;
namespace xsmt = mlir::xsmt;
namespace xsmt_async = mlir::xsmt_async;

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
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets) -> Value {
              auto AdvanceOp = self.create<triton::AdvanceOp>(base.getType(), base, offsets);
              auto LoadOp = self.create<triton::LoadOp>(AdvanceOp.getResult(), triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
             return LoadOp;
           })
      .def("create_descriptor_load_to_destination",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets, Value &destination) {
               auto AdvanceOp = self.create<triton::AdvanceOp>(base.getType(), base, offsets);
               auto LoadOp = self.create<triton::LoadOp>(AdvanceOp.getResult(), triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
               auto resultType = dyn_cast<RankedTensorType>(LoadOp.getType());
               int rank = resultType.getRank();
               std::vector<int32_t> boundary_check;
               for (int i = 0; i < rank; ++i) {
                  boundary_check.push_back(i);
               }
               self.create<mlir::triton::StoreOp>(destination, LoadOp,
                                           boundary_check,
                                           triton::CacheModifier::NONE,
                                           triton::EvictionPolicy::NORMAL);
           })
      .def("create_view",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &offsets,
              std::vector<int32_t> &shape,
              std::vector<int32_t> &packed_size) -> Value {
             return self.create<xsmt::ViewOp>(base, offsets, shape, packed_size);
           })
      .def("create_alloc",
           [](TritonOpBuilder &self, std::vector<int32_t> &shape,
              mlir::Type type, std::string storage) -> Value {
            if (shape.empty()) {
              throw std::runtime_error("alloc shape cannot be empty");
            }
            auto loc = self.getBuilder().getUnknownLoc();
            auto op = self.create<xsmt::AllocOp>(type, shape, storage);
            return op;
           })
      .def("create_mmt4d",
     [](TritonOpBuilder &self, Value &a, Value &b, std::optional<Value> c = std::nullopt) -> Value {
       auto aType = cast<RankedTensorType>(a.getType());
       auto bType = cast<RankedTensorType>(b.getType());

       assert(aType.getRank() == 4 && "A must be 4D packed tensor");
       assert(bType.getRank() == 4 && "B must be 4D packed tensor");

       auto aShape = aType.getShape();
       auto bShape = bType.getShape();
       assert(aShape[1] == bShape[0] && "KB dimension must match");
       assert(aShape[3] == bShape[2] && "kb dimension must match");

       SmallVector<int64_t> outputShape = {
           aShape[0], bShape[1], aShape[2], bShape[3],
       };
       auto resultType =
           RankedTensorType::get(outputShape, aType.getElementType());

       auto perm = std::vector<int>{1, 0, 3, 2};
       auto transbOp = self.create<mlir::triton::TransOp>(b, perm);
       mlir::Value transbValue = transbOp->getResult(0);

       mlir::Value cValue;
       if (c.has_value()) {
         cValue = *c;
       } else {
         cValue = Value();
       }

       return self.create<xsmt::MMT4DOp>(
           resultType,
           a,
           transbValue,
           cValue
       );
     })
      .def("create_mbarrier", [](TritonOpBuilder &self, Value &flag, Value &atc,
                            Value &tc, Value &exp) -> Value {
         auto barrierType = self.getBuilder().getI64Type();
         return self.create<mlir::xsmt_async::MBarrierAllocOp>(barrierType, flag, atc, tc, exp);
       })
      .def("create_barrier_arrive", [](TritonOpBuilder &self, Value &bar) {
             self.create<mlir::xsmt_async::MBarrierArriveOp>(bar);
           })
      .def("create_barrier_wait", [](TritonOpBuilder &self, Value &bar,
                                    Value &flag, Value &exp) {
             self.create<mlir::xsmt_async::MBarrierWaitOp>(bar, flag, exp);
           })
      .def("create_get_num_of_thread", [](TritonOpBuilder &self) {
             self.create<xsmt::GetThreadOp>();
           })
      .def("create_global_mbarrier", [](TritonOpBuilder &self, Value &id) -> Value{
             auto barrierType = self.getBuilder().getI64Type();
             return self.create<xsmt::GlobalMBarrierInitOp>(barrierType, id);
           })
      .def("create_barrier_set_expect", [](TritonOpBuilder &self, Value &bar, Value &exp) {
             self.create<xsmt::BarrierSetEepectOp>(bar, exp);
           });
}
void init_triton_spine_triton(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::xsmt::XSMTDialect, mlir::xsmt_async::XSMTAsyncDialect, tensor::TensorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_xsmt_ir(m.def_submodule("xsmt_ir"));
}
