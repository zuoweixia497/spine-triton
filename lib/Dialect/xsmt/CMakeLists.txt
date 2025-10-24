add_triton_library(XSMTIR
  XSMTDialect.cpp
  XSMTOps.cpp

  DEPENDS
  XSMTTableGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
)
