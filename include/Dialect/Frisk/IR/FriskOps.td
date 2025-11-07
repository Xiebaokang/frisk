#ifndef FRISK_OPS
#define FRISK_OPS

include "Dialect/Frisk/IR/FriskDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike

include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultRank


// Op Base
class Frisk_Op<string mnemonic, list<Trait> traits = []> : 
  Op<Frisk_Dialect, mnemonic, traits>;


def Frisk_GemmOp: Frisk_Op<"gemm", [Pure, InferTypeOpAdaptor]> {
  let summary = "General matrix multiplication operation";
  let description = [{}];
  let arguments = (ins
    AnyMemRef:$A,
    AnyMemRef:$B,
    AnyMemRef:$C
  );
  let assemblyFormat = "$A `,` $B `,` $C attr-dict `:` type(operands)" ;
}

// def Frisk_CopyOp: Frisk_Op<"copy", [Pure, InferTypeOpAdaptor]> {
//   let summary = "copy";

//   let description = [{}];

//   let arguments = (ins 
//     AnyRankedTensor:$src,
//     AnyRankedTensor:$dst
//   );

//   // let assemblyFormat = "$inputs attr-dict `:` functional-type(operands, results)";
// }

#endif // FRISK_OPS