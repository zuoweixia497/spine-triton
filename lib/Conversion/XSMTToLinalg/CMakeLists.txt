get_filename_component(PROJECT_REPO_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)

message(STATUS "Looking for backend lib in: ${PROJECT_REPO_ROOT}/backend/lib")

add_library(MLIRXSMTAsyncDialect STATIC IMPORTED)

set_target_properties(MLIRXSMTAsyncDialect PROPERTIES
  IMPORTED_LOCATION "${PROJECT_REPO_ROOT}/backend/lib/libMLIRXSMTAsyncDialect.a"
)

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
  MLIRXSMTAsyncDialect
)
