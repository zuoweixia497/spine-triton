set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS XSMTDialect.td)
mlir_tablegen(XSMTDialect.h.inc -gen-dialect-decls -dialect=xsmt)
mlir_tablegen(XSMTDialect.cpp.inc -gen-dialect-defs -dialect=xsmt)

set(LLVM_TARGET_DEFINITIONS XSMTOps.td)
mlir_tablegen(XSMTOps.h.inc -gen-op-decls)
mlir_tablegen(XSMTOps.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(XSMTTableGen)
