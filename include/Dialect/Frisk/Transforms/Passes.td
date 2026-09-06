#ifndef FRISK_PASSES
#define FRISK_PASSES

include "mlir/Pass/PassBase.td"

def FriskInferLayouts : Pass<"frisk-infer-layouts", "::mlir::ModuleOp"> {
  let summary = "Infer and materialize Frisk layouts";
  let constructor = "mlir::frisk::createFriskInferLayoutsPass()";
  let dependentDialects = ["::mlir::frisk::FriskDialect"];
}

#endif // FRISK_PASSES
