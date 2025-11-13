// ViewOpInfoStorage.h
#ifndef VIEW_OP_INFO_STORAGE_H
#define VIEW_OP_INFO_STORAGE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

struct ViewOpInfo {
  llvm::SmallVector<mlir::Value, 2> offsets;
  llvm::SmallVector<int32_t, 2> shape;
  llvm::SmallVector<int32_t, 2> microSize;
  mlir::Operation* viewOp;
};

class ViewOpInfoStorage {
private:
  llvm::DenseMap<mlir::Operation*, ViewOpInfo> descriptorToInfoMap;

public:
  void storeViewOpInfo(mlir::Operation* descriptorOp,
                      mlir::ValueRange offsets,
                      llvm::ArrayRef<int32_t> shape,
                      llvm::ArrayRef<int32_t> microSize,
                      mlir::Operation* viewOp) {
    ViewOpInfo info;
    info.offsets.assign(offsets.begin(), offsets.end());
    info.shape.assign(shape.begin(), shape.end());
    info.microSize.assign(microSize.begin(), microSize.end());

    descriptorToInfoMap[descriptorOp] = info;
  }

  bool getInfoForDescriptor(mlir::Operation* descriptorOp,
                           llvm::SmallVectorImpl<mlir::Value>* offsets,
                           llvm::SmallVectorImpl<int32_t>* shape,
                           llvm::SmallVectorImpl<int32_t>* microSize,
                           mlir::Operation** viewOp = nullptr) {
    auto it = descriptorToInfoMap.find(descriptorOp);
    if (it == descriptorToInfoMap.end()) return false;

    const ViewOpInfo& info = it->second;
    if (offsets) offsets->assign(info.offsets.begin(), info.offsets.end());
    if (shape) shape->assign(info.shape.begin(), info.shape.end());
    if (microSize) microSize->assign(info.microSize.begin(), info.microSize.end());
    if (viewOp) *viewOp = info.viewOp;

    return true;
  }

  void clear() {
    descriptorToInfoMap.clear();
  }

  bool contains(mlir::Operation* descriptorOp) const {
    return descriptorToInfoMap.find(descriptorOp) != descriptorToInfoMap.end();
  }
};

#endif // VIEW_OP_INFO_STORAGE_H