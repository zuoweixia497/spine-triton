get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(spine-triton-opt spine-triton-opt.cpp PARTIAL_SOURCES_INTENDED)

get_filename_component(PROJECT_REPO_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)

message(STATUS "Looking for backend lib in: ${PROJECT_REPO_ROOT}/backend/lib")

add_library(MLIRXSMTAsyncDialect STATIC IMPORTED)

set_target_properties(MLIRXSMTAsyncDialect PROPERTIES
  IMPORTED_LOCATION "${PROJECT_REPO_ROOT}/backend/lib/libMLIRXSMTAsyncDialect.a"
)

# TODO: what's this?
llvm_update_compile_flags(spine-triton-opt)
target_link_libraries(spine-triton-opt PRIVATE
  TritonTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRXSMTAsyncDialect
)

mlir_check_all_link_libraries(spine-triton-opt)
