#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#
get_filename_component(PROJECT_REPO_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)

message(STATUS "Looking for backend lib in: ${PROJECT_REPO_ROOT}/backend/lib")

add_library(MLIRXSMTAsyncDialect STATIC IMPORTED)

set_target_properties(MLIRXSMTAsyncDialect PROPERTIES
  IMPORTED_LOCATION "${PROJECT_REPO_ROOT}/backend/lib/libMLIRXSMTAsyncDialect.a"
)

add_triton_library(TritonToLinalgExperimental
  TritonToLinalgExperimentalPass.cpp
  ReconcilePtrCastsPass.cpp
  ReconcileLlvmPtrCastsPass.cpp
  TritonToPtrPass.cpp
  ScfbufferStandardizedPass.cpp
  ConvertScanOpPass.cpp
  #CollapseShape.cpp

  DEPENDS
  TritonToLinalgExperimentalConversionPassIncGen

  LINK_LIBS PUBLIC
  TritonTilingExtIR
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TPtrIR
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
  TritonSharedUtils

  TritonArithToLinalg
  StructuredToMemref
  TritonToStructured
  TritonToUnstructured
  TritonPtrToMemref
  UnstructuredToMemref
  AddTargetDescription
  XSMTToLinalg
  MLIRXSMTAsyncDialect
)
