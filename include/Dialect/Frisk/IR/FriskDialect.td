#ifndef FRISK_DIALECT
#define FRISK_DIALECT

include "mlir/IR/OpBase.td"


def Frisk_Dialect : Dialect {
  let name = "frisk";

  let cppNamespace = "::mlir::frisk";

  let summary = "Frisk Dialact in MLIR";

  let description = [{
    Frisk Dialect.
  }];

  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
}

#endif // FRISK_DIALECT