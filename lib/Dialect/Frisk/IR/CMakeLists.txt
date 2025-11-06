add_frisk_library(FriskIR
  FriskOps.cpp

  DEPENDS
  FriskTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)