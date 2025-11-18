#ifndef FRISK_OPS
#define FRISK_OPS

include "Dialect/Frisk/IR/FriskDialect.td"
include "Dialect/Frisk/IR/FriskEnums.td"
include "Dialect/Frisk/IR/FriskAttributes.td"
// include "Dialect/Frisk/IR/FriskInterfaces.td"

include "mlir/IR/OpBase.td"
// include "mlir/IR/AttrTypeBase.td"
// include "mlir/IR/EnumAttr.td"
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

def ImplicitFriskTerminator : SingleBlockImplicitTerminator<"EndOp">;

// KernelOp
def Frisk_KernelOp : Frisk_Op<"kernel", [ImplicitFriskTerminator]> {
    let summary = "Kernel operation";
  let description = [{
    A simple kernel operation that represents a compute kernel.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$function_type
  );

  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type)>
  ];

  let extraClassDeclaration = [{
    unsigned getNumArguments() {
      return getBody(0)->getNumArguments();
    }

    BlockArgument getArgument(unsigned idx) {
      return getBody(0)->getArgument(idx);
    }

    Block* addEntryBlock() {
      auto& region = getRegion();
      if (!region.empty()) {
        llvm_unreachable("entry block already exists");
      }
      auto* entry = new Block;
      region.push_back(entry);
      // block add args
      mlir::Type funcType = getFunctionType();
      if (auto funcTy = mlir::dyn_cast<FunctionType>(funcType)) {
        for (auto type : funcTy.getInputs()) {
          entry->addArgument(type, getLoc());
        }
      }
      mlir::OpBuilder builder(entry, entry->begin());
      builder.create<EndOp>(getLoc());
      return entry;
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// parallelOp
def Frisk_ParallelOp : Frisk_Op<"parallel", [Pure, HasParent<"KernelOp">, ImplicitFriskTerminator]> {
  let summary = "parallel opeartion";
  let description = [{
    parallel
  }];

  let arguments = (ins
    I64Attr:$threads,
    DenseI64ArrayAttr:$ranges
  );

  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$ranges, "int64_t":$thread_num)>
  ];

  let extraClassDeclaration = [{
    std::vector<int64_t> getGrid() {
      std::vector<int64_t> ranges;
      for (auto range : getRangesAttr().asArrayRef()) {
        ranges.push_back(range);
      }
      return ranges;
    }

    unsigned getGridDims() { 
      return getRangesAttr().size(); 
    }

    int64_t getThreadNum() {
      return getThreads();
    }

    MutableArrayRef<BlockArgument> getIVs() {
      return getBody()->getArguments();
    }

    Block* addEntryBlock() {
      auto& region = getRegion();
      if (!region.empty()) {
        llvm_unreachable("entry block already exists");
      }
      auto* entry = new Block;
      region.push_back(entry);
      for (unsigned i=0; i<getGridDims(); i++) {
        entry->addArgument(IndexType::get(getContext()), getLoc());
      }
      mlir::OpBuilder builder(entry, entry->begin());
      builder.create<EndOp>(getLoc());
      return entry;
    }
  }];

  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 1;
}

// blockOp
def Frisk_BlockOp : Frisk_Op<"block", [Pure, ImplicitFriskTerminator]> {
  let summary = "block opeartion";
  let description = [{
    block
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$ranges
  );

  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$ranges, 
               CArg<"function_ref<void(ValueRange)>", "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn = function_ref<void(ValueRange)>;

    std::vector<int64_t> getBlockRanges() {
      std::vector<int64_t> ranges;
      for (auto range : getRangesAttr().asArrayRef()) {
        ranges.push_back(range);
      }
      return ranges;
    }

    int64_t getIterationNum() {
      int64_t iter_num = 1;
      for (int64_t range : getBlockRanges()) {
        iter_num *= range;
      }
      return iter_num;
    }

    int64_t getBlockDim() {
      return getRangesAttr().size();
    }

    MutableArrayRef<BlockArgument> getIVs() {
      return getBody()->getArguments();
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// forOp
def Frisk_ForOp : Frisk_Op<"for", [Pure, ImplicitFriskTerminator]> {
  let summary = "for opeartion";
  let description = [{
    for
  }];

  let arguments = (ins I64Attr:$lower, I64Attr:$upper, I64Attr:$step);

  let regions = (region SizedRegion<1>:$region);

  let builders = [OpBuilder<(ins "int64_t":$lowerBound, 
                  "int64_t":$upperBound,
              CArg<"int64_t", "1">:$step, 
              CArg<"function_ref<void(Value)>", "nullptr">:$bodyBuilder)>];
  
  let extraClassDeclaration = [{
    using BodyBuilderFn = function_ref<void(Value)>;

    BlockArgument getInductionVar() { return getBody()->getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front();
    }

    // /// Returns information about the lower bound as a single object.
    // int64_t getLowerBound() { return getProperties().lower; }

    // /// Returns information about the upper bound as a single object.
    // int64_t getUpperBound() { return getProperties().upper; }

    // /// Returns loop step.
    // int64_t getStep() { return getProperties().step; }
  }];

  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 1;
}

// EndOp
def Frisk_EndOp : Frisk_Op<"end", [Terminator, ReturnLike]> {
  let summary = "Invisible kernel termination operation";
  let description = [{
    An invisible terminator operation for frisk kernels that dose not appear in printed IR.
  }];

  let arguments = (ins);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  let hasVerifier = 0;
  let assemblyFormat = "attr-dict";
}

// gemmOp
def Frisk_GemmOp: Frisk_Op<"gemm", [Pure]> {
  let summary = "General matrix multiplication operation";
  let description = [{}];
  let arguments = (ins
    AnyMemRef:$A,
    AnyMemRef:$B,
    AnyMemRef:$C,

    BoolAttr:$transA,
    BoolAttr:$transB,
    I64Attr:$M,
    I64Attr:$N,
    I64Attr:$K,
    GemmWarpPolicyAttr:$policy,
    BoolAttr:$clear_accum
  );

  let results = (outs);

  let extraClassDeclaration = [{
    Value getMatrixA() { return getA(); }
    Value getMatrixB() { return getB(); }
    Value getMatrixC() { return getC(); }
    ValueRange getAllMatrix() {
      return ValueRange(getOperation()->getOperands());
    }
  }];

  // let extraClassDeclaration = [{
  //   Value getMatrixA() { return getOperand(0); }
  //   Value getMatrixB() { return getOperand(1); }
  //   Value getMatrixC() { return getOperand(2); }
  //   ValueRange getAllMatrix() {
  //     return getOperation()->getOperands();
  //   }
  // }];

  let hasVerifier = 1;
  let assemblyFormat = "`(` $A `,` $B `,` $C `)` attr-dict `:` type($A) `,` type($B) `,` type($C)";
}

// allocOp
def Frisk_AllocBufferOp: Frisk_Op<"alloc_buffer", [Pure]> {
  let summary = "allocate buffer operation";
  let description = [{ 
    Allocates a buffer with the specified shape, element type and optional attributes.
  }];

  let arguments = (ins 
    DenseI64ArrayAttr:$shape,
    TypeAttr:$elementType,
    DefaultValuedAttr<I64Attr, "0">:$alignment,
    // DefaultValuedAttr<I64Attr, "0">:$memorySpace
    DefaultValuedAttr<MemorySpaceAttr, "::mlir::frisk::attr::MemorySpace::Local">:$memorySpace
  );

  let results = (outs 
    Res<AnyMemRef>:$result
  );

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
    // OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "int64_t":$alignment)>,
    // OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "int64_t":$alignment, "int64_t":$memorySpace)>
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "Attribute":$memorySpace)>
  ];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() { 
      return cast<MemRefType>(getResult().getType()); 
    }
    // ArrayRef<int64_t> getShape() { return getShapeAttr().asArrayRef(); }
    // Type getElementType() { return getElementTypeAttr().getValue(); }
    // int64_t getAlignment() { return getAlignmentAttr().getInt(); }
    // int64_t getMemorySpace() { return getMemorySpaceAttr().getInt(); }
    
    // 根据 memorySpace 获取 scope
    StringRef getScope() {
      switch (getMemorySpace()) {
        case ::mlir::frisk::attr::MemorySpace::Local:
          return "local";
        case ::mlir::frisk::attr::MemorySpace::Global:
          return "global";
        case ::mlir::frisk::attr::MemorySpace::Shared:
          return "shared";
        // case 0: return "local";
        // case 1: return "global"; 
        // case 3: return "shared";
        // default: return "unknown";
      }
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// copyOp
def Frisk_CopyOp: Frisk_Op<"copy", [Pure, AttrSizedOperandSegments]> {
  let summary = "copy";
  let description = [{ 
    copy
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "", [MemRead]>:$src,
    Arg<AnyMemRef, "", [MemWrite]>:$dst,
    AffineMapAttr:$srcMap,
    AffineMapAttr:$dstMap,
    Variadic<Index>:$srcIndices,
    Variadic<Index>:$dstIndices,
    DenseI64ArrayAttr:$srcExtents,
    DenseI64ArrayAttr:$dstExtents
  );

  let builders = [
    OpBuilder<(ins "Value":$srcMemref, "Value":$dstMemref, 
      "ValueRange":$srcIndices, "ValueRange":$dstIndices)>,

    OpBuilder<(ins "Value":$srcMemref, "Value":$dstMemref, 
      "AffineMap":$srcMap, "AffineMap":$dstMap,
      "ValueRange":$srcIndices, "ValueRange":$dstIndices)>
  ];

  let extraClassDeclaration = [{
    Value getSrcMemRef() { return getOperand(0); }
    Value getDstMemRef() { return getOperand(1); }

    MemRefType getSrcMemRefType() { 
      return cast<MemRefType>(getSrcMemRef().getType()); 
    }
    MemRefType getDstMemRefType() { 
      return cast<MemRefType>(getDstMemRef().getType()); 
    }

  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// fillOp
def Frisk_FillOp: Frisk_Op<"fill", [Pure]> {
  let summary = "fill";
  let description = [{ 
    fill
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "", [MemWrite]>:$memref,
    AnyAttr:$value
  );

  let extraClassDeclaration = [{
    MemRefType getMemType() { return cast<MemRefType>(getMemref().getType()); }
  }];

  let hasVerifier = 1;
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

// reduceOp
def Frisk_ReduceOp: Frisk_Op<"reduce", [Pure]> {
  let summary = "reduce";
  let description = [{ 
    reduce
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "", [MemRead]>:$src,
    Arg<AnyMemRef, "", [MemWrite]>:$dst,
    StrAttr:$kind,
    I64Attr:$dim,
    BoolAttr:$clear
  );

  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$dst, 
               "StringRef":$kind, "int64_t":$dim, "bool":$clear)>
  ];

  let extraClassDeclaration = [{
    MemRefType getSrcType() { return cast<MemRefType>(getSrc().getType()); }
    MemRefType getDstType() { return cast<MemRefType>(getDst().getType()); }
  }];

  let hasVerifier = 1;
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

#endif // FRISK_OPS