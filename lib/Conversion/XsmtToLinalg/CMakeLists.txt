add_triton_library(XsmtToLinalg
  XsmtToLinalg.cpp
  XsmtToLinalgPass.cpp

  DEPENDS
  XsmtToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgTransforms
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonTilingExtIR
  TritonStructuredIR
  TritonSharedUtils
)
