#ifndef FRISK_PASSES
#define FRISK_PASSES

include "mlir/Pass/PassBase.td"

def FriskInferLayouts : Pass<"frisk-infer-layouts", "::mlir::ModuleOp"> {
  let summary = "Infer and materialize Frisk layouts";
  let constructor = "mlir::frisk::createFriskInferLayoutsPass()";
  let dependentDialects = ["::mlir::frisk::FriskDialect"];
  let options = [
    Option<"analysisOnly", "analysis-only", "bool", "false",
           "Collect, solve and verify layouts without modifying IR">,
    Option<"dumpAnalysis", "dump-analysis", "bool", "false",
           "Print the deterministic graph, domain sizes and conversion decisions">
  ];
}

#endif // FRISK_PASSES
