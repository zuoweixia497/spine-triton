add_triton_library(TLXIR
  SMTDialect.cpp
  SMTOps.cpp

  DEPENDS
  SMTTableGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
)
