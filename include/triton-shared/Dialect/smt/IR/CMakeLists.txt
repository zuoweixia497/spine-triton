set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS SMTDialect.td)
mlir_tablegen(SMTDialect.h.inc -gen-dialect-decls -dialect=smt)
mlir_tablegen(SMTDialect.cpp.inc -gen-dialect-defs -dialect=smt)

set(LLVM_TARGET_DEFINITIONS SMTOps.td)
mlir_tablegen(SMTOps.h.inc -gen-op-decls)
mlir_tablegen(SMTOps.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(SMTTableGen)
