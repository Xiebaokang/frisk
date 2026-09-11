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

def OptimizeLayoutConversions : Pass<"frisk-optimize-layout-conversions", "::mlir::ModuleOp"> {
  let summary = "Fold and compose type-safe distributed layout conversions";
  let constructor = "mlir::frisk::createOptimizeLayoutConversionsPass()";
}

def TestLowerLayoutConversions : Pass<"frisk-test-lower-layout-conversions", "::mlir::ModuleOp"> {
  let summary = "Test-only static single-CTA layout redistribution adapter";
  let constructor = "mlir::frisk::createTestLowerLayoutConversionsPass()";
  let options = [Option<"scratchBudget", "scratch-budget", "uint64_t", "49152",
    "Conservative adapter workgroup byte budget (not the SM90 maximum)">];
}

#endif // FRISK_PASSES
