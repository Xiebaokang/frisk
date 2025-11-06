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
  let summary = "gemm";
  let description = [{}];
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs AnyRankedTensor:$result);
  // let assemblyFormat = "$operand `,` `type` `=` $dst_type attr-dict `:` functional-type(operands, results)";
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