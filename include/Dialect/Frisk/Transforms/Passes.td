#ifndef FRISK_PASSES
#define FRISK_PASSES

include "mlir/Pass/PassBase.td"

def FriskLayoutInfer: InterfacePass<"frisk-layoutInfer", "FunctionOpInterface">{
    let summary = "Frisk layoutInfer";
    let description = [{
        Infer layout op
    }];

    let constructor = "mlir::frisk::createFriskLayoutInferPass";

    let dependentDialects = [];

}

#endif // FRISK_PASSES