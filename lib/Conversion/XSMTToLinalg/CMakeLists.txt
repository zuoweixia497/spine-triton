add_triton_library(XSMTToLinalg
  XSMTToLinalg.cpp
  XSMTToLinalgPass.cpp

  DEPENDS
  XSMTToLinalgConversionPassIncGen

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
