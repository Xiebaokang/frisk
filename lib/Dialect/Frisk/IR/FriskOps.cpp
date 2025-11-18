#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
// #include "Dialect/Frisk/IR/FriskInterfaces.h"

// #include "Dialect/Frisk/IR/FriskEnums.cpp.inc"
#define GET_OP_CLASSES
#include "Dialect/Frisk/IR/FriskOps.cpp.inc"

// move dialect def in this file to make compiler happy
#include "Dialect/Frisk/IR/FriskDialect.cpp.inc"
namespace mlir {
namespace frisk {

} // namespace frisk
} // namespace mlir

namespace mlir {
namespace frisk {

//===----------------------------------------------------------------------===//
// -- KernelOp --
//===----------------------------------------------------------------------===//
void KernelOp::build(OpBuilder &builder, OperationState &state, StringRef name, FunctionType type) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addRegion();
}

LogicalResult KernelOp::verify() {
  auto typeAttr = getFunctionTypeAttr();
  if (!typeAttr)
    return emitOpError("requires a 'function_type' attribute");

  auto functionType = llvm::dyn_cast<FunctionType>(typeAttr.getValue());
  if (!functionType)
    return emitOpError("requires a function type");

  // 验证区域参数
  if (getBody(0)->getNumArguments() != functionType.getNumInputs())
    return emitOpError("region argument count does not match function type");

  for (unsigned i = 0; i < getBody(0)->getNumArguments(); ++i) {
    if (getBody(0)->getArgument(i).getType() != functionType.getInput(i))
      return emitOpError("region argument type mismatch");
  }

  return success();
}

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析符号名称
  StringAttr symName;
  if (parser.parseSymbolName(symName, "sym_name", result.attributes))
    return failure();
  // 解析参数列表
  SmallVector<Type> argTypes;
  if (parser.parseLParen() || parser.parseTypeList(argTypes) || parser.parseRParen())
    return failure();
  // 解析结果类型 - 可选，如果没有结果就是空
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseLParen() || parser.parseTypeList(resultTypes) || parser.parseRParen())
      return failure();
  }
  // 创建函数类型属性
  auto functionType = parser.getBuilder().getFunctionType(argTypes, resultTypes);
  result.addAttribute("function_type", TypeAttr::get(functionType));
  // 解析区域
  Region *body = result.addRegion();
  SmallVector<OpAsmParser::Argument> args;
  for (Type argType : argTypes) {
    args.emplace_back();
    args.back().type = argType;
  }
  if (parser.parseRegion(*body, args) || 
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void KernelOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(getSymName());
  // 打印参数类型
  auto funcType = getFunctionType();
  auto funcTy = dyn_cast<FunctionType>(funcType);
  p << "(";
  llvm::interleaveComma(funcTy.getInputs(), p);
  p << ")";
  // 只有当有结果时才打印结果类型
  if (!funcTy.getResults().empty()) {
    p << " -> (";
    llvm::interleaveComma(funcTy.getResults(), p);
    p << ")";
  }
  // 打印区域
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
  // 打印属性
  p.printOptionalAttrDict((*this)->getAttrs(), {"sym_name", "function_type"});
}

//===----------------------------------------------------------------------===//
// -- ParallelOp --
//===----------------------------------------------------------------------===//
void ParallelOp::build(OpBuilder &builder, OperationState &state,
                      llvm::ArrayRef<int64_t> ranges, int64_t thread_num) {
  state.addAttribute("threads", builder.getI64IntegerAttr(thread_num));
  state.addAttribute("ranges", builder.getDenseI64ArrayAttr(ranges));
  state.addRegion();
  // Region *region = state.regions[0].get();
  // Block *entry = new Block();
  // region->push_back(entry);
  // for (unsigned i=0; i<ranges.size(); ++i) {
  //   entry->addArgument(builder.getIndexType(), state.location);
  // }
}

ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析迭代变量列表: (%arg1, %arg2)
  SmallVector<OpAsmParser::Argument, 4> inductionVars;
  if (parser.parseArgumentList(inductionVars, AsmParser::Delimiter::Paren))
    return failure();
  // 解析等号和范围: = (8, 8)
  if (parser.parseEqual() || parser.parseLParen())
    return failure();
  SmallVector<int64_t> ranges;
  if (parser.parseCommaSeparatedList([&]() {
        int64_t range;
        if (parser.parseInteger(range))
          return failure();
        ranges.push_back(range);
        return success();
      }) || parser.parseRParen())
    return failure();
  // 解析线程数量: , threads=128
  int64_t threads = 0;
  if (parser.parseComma() || parser.parseKeyword("threads") || 
      parser.parseEqual() || parser.parseInteger(threads))
    return failure();
  // 添加属性
  result.addAttribute("ranges", parser.getBuilder().getDenseI64ArrayAttr(ranges));
  result.addAttribute("threads", parser.getBuilder().getI64IntegerAttr(threads));
  // 解析区域
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVars))
    return failure();
  return success();
}

void ParallelOp::print(OpAsmPrinter &p) {
  // 打印迭代变量: (%arg1, %arg2)
  p << " (";
  llvm::interleaveComma(getBody(0)->getArguments(), p, [&](BlockArgument arg) {
    p << arg;
  });
  p << ")";
  // 打印范围和线程配置: = (8, 8), threads=128
  p << " = (";
  auto grid = getGrid();
  llvm::interleaveComma(grid, p);
  p << "), threads = " << getThreadNum();
  // 打印区域（不打印终止符）
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// -- BlockOp --
//===----------------------------------------------------------------------===//
void BlockOp::build(OpBuilder &builder, OperationState &state, ArrayRef<int64_t> ranges, BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);
  
  state.addAttribute("ranges", builder.getDenseI64ArrayAttr(ranges));

  Region *bodyRegion = state.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  llvm::SmallVector<Value> inductionVars;
  for (int64_t range : ranges) {
    inductionVars.push_back(bodyBlock->addArgument(builder.getIndexType(), state.location));
  }
  if (!bodyBuilder) {
    ensureTerminator(*bodyRegion, builder, state.location);
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(inductionVars);
    builder.create<EndOp>(state.location);
  }
}

ParseResult BlockOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析参数列表: (%arg3, %arg4)
  SmallVector<OpAsmParser::Argument, 4> blockArgs;
  if (parser.parseArgumentList(blockArgs, OpAsmParser::Delimiter::Paren))  // 解析 "to" 关键字
    return failure();
  if (parser.parseKeyword("to"))  // 解析边界列表: (128, 128)
    return failure();
  SmallVector<int64_t, 4> ranges;
  if (parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        [&]() -> ParseResult {
          int64_t range;
          if (parser.parseInteger(range))
            return failure();
          ranges.push_back(range);
          return success();
        }))
    return failure();
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs))  // 解析区域 { ... }
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))  // 解析可选的属性字典
    return failure();
  if (!ranges.empty()) {  // 将边界添加到属性中
    result.addAttribute("ranges", parser.getBuilder().getDenseI64ArrayAttr(ranges));
  }
  return success();
}

void BlockOp::print(OpAsmPrinter &p) {
  // 打印迭代变量: (%arg1, %arg2)
  p << " (";
  llvm::interleaveComma(getBody(0)->getArguments(), p, [&](BlockArgument arg) {
    p << arg;
  });
  p << ")";
  // 打印范围和线程配置: = (8, 8), threads=128
  p << " to (";
  auto ranges = getBlockRanges();
  llvm::interleaveComma(ranges, p);
  p << ")";
  // 打印区域（不打印终止符）
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

LogicalResult BlockOp::verify() {   // 暂时关闭block op的验证方法
  // auto &blockBody = getRegion();
  // if (blockBody.empty()) {
  //   return emitOpError("block must have a body");
  // }
  // auto &block = blockBody.front();
  // bool hasAffineLoadOp = false;
  // bool hasAffineStoreOp = false;
  
  // for (auto &op : block.getOperations()) {
  //   if (isa<affine::AffineLoadOp>(op)) {    // 检查是否是affine.load操作
  //     hasAffineLoadOp = true;
  //   }
  //   else if (isa<affine::AffineStoreOp>(op)) {    // 检查是否是affine.store操作  
  //     hasAffineStoreOp = true;
  //   }
  //   // 如果已经找到两种操作，可以提前退出
  //   if (hasAffineLoadOp && hasAffineStoreOp) {
  //     break;
  //   }
  // }
  // // 验证结果
  // if (!hasAffineLoadOp) {
  //   return emitOpError("block must contain at least one affine.load operation");
  // }
  // if (!hasAffineStoreOp) {
  //   return emitOpError("block must contain at least one affine.store operation");
  // }
  return success();
}

//===----------------------------------------------------------------------===//
// -- ForOp --
//===----------------------------------------------------------------------===//
void ForOp::build(OpBuilder &builder, OperationState &state, 
                  int64_t lowerBound, int64_t upperBound, int64_t step, 
                  BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  state.addAttribute("lower", builder.getI64IntegerAttr(lowerBound));
  state.addAttribute("upper", builder.getI64IntegerAttr(upperBound));
  state.addAttribute("step", builder.getI64IntegerAttr(step));

  Region *bodyRegion = state.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  auto iv = bodyBlock->addArgument(builder.getIndexType(), state.location);
  if (!bodyBuilder) {
    ensureTerminator(*bodyRegion, builder, state.location);
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(iv);
    builder.create<EndOp>(state.location);
  }
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析格式：frisk.for %arg1 = 0 to 1024 step = 32 { ... }
  OpAsmParser::UnresolvedOperand inductionVar;
  IntegerAttr lowerAttr, upperAttr, stepAttr;

  if (parser.parseOperand(inductionVar) || parser.parseEqual())  // 解析循环变量 "%arg1 ="
    return failure();
  auto builder = parser.getBuilder();
  if (parser.parseAttribute(lowerAttr, builder.getIntegerType(64), "lower", result.attributes))  // 解析下界
    return failure();
  if (parser.parseKeyword("to"))  // 解析 "to"
    return failure();
  if (parser.parseAttribute(upperAttr, builder.getIntegerType(64), "upper", result.attributes))  // 解析上界
    return failure();
  if (parser.parseKeyword("step") || parser.parseEqual())  // 解析 "step ="
    return failure();
  if (parser.parseAttribute(stepAttr, builder.getIntegerType(64), "step", result.attributes))  // 解析步长
    return failure();

  std::unique_ptr<Region> region = std::make_unique<Region>();  // 解析区域
  llvm::SmallVector<OpAsmParser::Argument, 4> regionArgs;  // 解析区域参数类型（循环变量）
  llvm::SmallVector<Type, 4> regionArgTypes;

  regionArgTypes.push_back(builder.getIndexType());  // 循环变量类型为index
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren, true) || parser.parseRegion(*region, regionArgs))
    return failure();
  result.addRegion(std::move(region));
  return success();
}

void ForOp::print(OpAsmPrinter &p) {
  p << " ";
  p << getInductionVar() << " = " << getLower();  // 打印循环变量 //打印下界
  p << " to " << getUpper(); // 打印上界
  if (getStep() > 1) {
    p << " step = " << getStep();  // 打印步长
  }
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// -- GemmOp --
//===----------------------------------------------------------------------===//
LogicalResult GemmOp::verify() {
  auto AType = dyn_cast<MemRefType>(getA().getType());
  auto BType = dyn_cast<MemRefType>(getB().getType());
  auto CType = dyn_cast<MemRefType>(getC().getType());
  // 基础类型检查
  if (AType.getElementType() != BType.getElementType() ||
      AType.getElementType() != CType.getElementType()) {
      return emitOpError("all operands must have the same element type");
  }
  // 维度兼容性检查
  if (AType.getDimSize(1) != BType.getDimSize(0)) {
      return emitOpError("A columns must equal B rows for matrix multiplication");
  }
  // 输出维度检查
  if (AType.getDimSize(0) != CType.getDimSize(0) ||
      BType.getDimSize(1) != CType.getDimSize(1)) {
      return emitOpError("C dimensions must match matrix multiplication result");
  }
  return success();
}

// LogicalResult GemmOp::inferLayout(OpBuilder &builder, DenseMap<Value, Attribute> &LayoutMap) {
//   Operation *op = this->getOperation();

//   Value A = getA(), B = getB(), C = getC();
//   bool transA = getTransA(), transB = getTransB();
//   int64_t M = getM(), N = getN(), K = getK();
//   auto policy = getPolicy();

//   // auto target = frisk::getTargetFromOp(op);
//   // int blockSize = frisk::getBlockSizeFrom(op);
//   if(!target || blockSize <= 0){
//     return op->emitError("Failed to get target or block size could not infer layout");
//   }

//   frisk::GemmInst gemmInst = frisk::GetGemmInst(op, blockSize, target);
//   auto [warpM, warpN] = frisk::ComputeWarpPartition(policy, M, N, blockSize, target, gemnInst);

//   Attribute layoutA, layoutB, fragC;
//   int elementSize = A.getType().cast<MemRefType>().getElementTypeBitWidth();
//   auto ctx = builder.getContext();

//   if(frisk::TargetIsHopper(target)){
//     auto shapeA = /* */;
//     auto fwdIndexA = /* */;
//     layoutA = LayoutAttr::get(ctx, shapeA, fwdIndexA, nullptr, nullptr);

//     auto shapeB = /* */;
//     auto fwdIndexB = /* */;
//     layoutB = LayoutAttr::get(ctx, shapeB, fwdIndexB, nullptr, nullptr);

//     auto shapeC = /* */;
//     auto fwdIndexC = /* */;
//     auto fwdThreadC = /* */;
//     auto repSizeC = /* */;
//     layoutC = LayoutAttr::get(ctx, shapeC, fwdIndexC, fwdThreadC, repSizeC);
//   }else if{

//   }else{

//   }

//   if(!layoutA || !layoutB || !fragC){
//     return op->emitError("Failed to infer layout attributes for current config");
//   }

//   bool updated = false;
//   auto itA = LayoutMap.find(A);
//   if(itA == LayoutMap.end()){
//     layoutMap[A] = layoutA;
//     updated = true;
//   }else{
//     //检查兼容性，合并或报错
//   }

//   auto itB = LayoutMap.find(B);
//   if(itB == LayoutMap.end()){
//     layoutMap[B] = layoutB;
//     updated = true;
//   }else{
//     //检查兼容性，合并或报错
//   }

//   auto itC = LayoutMap.find(C);
//   if(itC == LayoutMap.end()){
//     layoutMap[getC()] = layoutC;
//     updated = true;
//   }else{
//     //检查兼容性，合并或报错
//   }

//   return success(updated);
// }
//===----------------------------------------------------------------------===//
// -- AllocBufferOp --
//===----------------------------------------------------------------------===//
// void AllocBufferOp::build(OpBuilder &builder, OperationState &state,
//                           ArrayRef<int64_t> shape, Type elementType) {
//   build(builder, state, shape, elementType, /*alignment=*/0, /*memorySpace=*/0);
// }

// void AllocBufferOp::build(OpBuilder &builder, OperationState &state,
//                           ArrayRef<int64_t> shape, Type elementType, 
//                           int64_t alignment) {
//   build(builder, state, shape, elementType, alignment, /*memorySpace=*/0);
// }

// void AllocBufferOp::build(OpBuilder &builder, OperationState &state,
//                           ArrayRef<int64_t> shape, Type elementType,
//                           int64_t alignment, int64_t memorySpace) {
//   // 创建 memref 类型
//   auto memrefType = MemRefType::get(shape, elementType, /*layout=*/{}, memorySpace);
//   // 添加属性
//   state.addAttribute("shape", builder.getDenseI64ArrayAttr(shape));
//   state.addAttribute("elementType", TypeAttr::get(elementType));
//   state.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
//   state.addAttribute("memorySpace", builder.getI64IntegerAttr(memorySpace));
//   // 添加结果类型
//   state.addTypes(memrefType);
// }

LogicalResult AllocBufferOp::verify() {
  auto resultType = getResult().getType();
  
  // 检查结果类型是否是 memref
  if (!isa<MemRefType>(resultType)) {
    return emitOpError("result must be a memref type");
  }
  
  auto memrefType = cast<MemRefType>(resultType);
  // 检查属性与结果类型是否一致
  auto attrShape = getShape();
  auto attrElementType = getElementType();
  auto attrMemorySpace = getMemorySpace();
  if (memrefType.getShape() != attrShape) {
    return emitOpError("shape attribute must match result memref shape");
  }
  if (memrefType.getElementType() != attrElementType) {
    return emitOpError("elementType attribute must match result memref element type");
  }

  unsigned resultMemorySpace = memrefType.getMemorySpaceAsInt();
  if (resultMemorySpace != static_cast<unsigned>(attrMemorySpace)) {
    return emitOpError("memorySpace attribute must match result memref memory space");
  }
  // 检查对齐值是否有效
  int64_t alignment = getAlignment();
  if (alignment < 0) {
    return emitOpError("alignment must be non-negative");
  }
  // 检查对齐值是否是 2 的幂（可选，但推荐）
  if (alignment > 0 && (alignment & (alignment - 1)) != 0) {
    return emitOpError("alignment must be a power of 2");
  }
  // 检查 memorySpace 是否有效
  switch (attrMemorySpace) {
    case ::mlir::frisk::attr::MemorySpace::Local:
      break;
    case ::mlir::frisk::attr::MemorySpace::Global:
      break;
    case ::mlir::frisk::attr::MemorySpace::Shared:
      break;
    default:
      return emitOpError("memorySpace must be Local, Global, or Shared");
  }
  return success();
}

ParseResult AllocBufferOp::parse(OpAsmParser &parser, OperationState &result) {
  // 直接解析属性字典
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // 解析结果类型: -> memref<32x128xf32>
  if (parser.parseArrow())
    return failure();
  Type resultType;
  if (parser.parseType(resultType))
    return failure();
  // 检查结果类型是否是 memref
  if (!isa<MemRefType>(resultType)) {
    return parser.emitError(parser.getCurrentLocation(), "expected memref type for result");
  }
  auto memrefType = cast<MemRefType>(resultType);
  // 从结果类型提取 shape 和 elementType
  auto shape = memrefType.getShape();
  auto elementType = memrefType.getElementType();
  auto memorySpace = memrefType.getMemorySpace();
  // 添加结果类型
  result.addTypes(resultType);
  // 添加必要的属性（如果属性字典中没有）
  auto &builder = parser.getBuilder();
  if (!result.attributes.get("shape")) {
    result.addAttribute("shape", builder.getDenseI64ArrayAttr(shape));
  }
  if (!result.attributes.get("elementType")) {
    result.addAttribute("elementType", TypeAttr::get(elementType));
  }
  if (!result.attributes.get("memorySpace")) {
    result.addAttribute("memorySpace", memorySpace);
  }
  return success();
}

// 自定义汇编格式打印
void AllocBufferOp::print(OpAsmPrinter &p) {
  // 打印属性字典
  p << " {";
  auto memorySpace = getMemorySpace();
  p << "scope = \"";
  switch (memorySpace) {
    case ::mlir::frisk::attr::MemorySpace::Local:
      p << "local";
      break;
    case ::mlir::frisk::attr::MemorySpace::Global:
      p << "global";
      break;
    case ::mlir::frisk::attr::MemorySpace::Shared:
      p << "shared";
      break;
    default:
      p << "unknown";
      break;
  }
  p << "\"";
  // 打印 alignment（如果不是默认值）
  if (getAlignment() != 0)
    p << ", alignment = " << getAlignment();
  p << "}";
  // 打印结果类型
  p << " -> " << getResult().getType();
}


//===----------------------------------------------------------------------===//
// -- CopyOp --
//===----------------------------------------------------------------------===//
void CopyOp::build(OpBuilder &builder, OperationState &state, 
                  Value srcMemref, Value dstMemref, ValueRange srcIndices, ValueRange dstIndices) {
  auto srcMemrefType = llvm::cast<MemRefType>(srcMemref.getType());
  auto dstMemrefType = llvm::cast<MemRefType>(dstMemref.getType());
  int64_t srcRank = srcMemrefType.getRank();
  int64_t dstRank = dstMemrefType.getRank();
  // Create identity map for memrefs with at least one dimension or () -> ()
  // for zero-dimensional memrefs.
  auto srcMap = srcRank ? builder.getMultiDimIdentityMap(srcRank) : builder.getEmptyAffineMap();
  auto dstMap = dstRank ? builder.getMultiDimIdentityMap(dstRank) : builder.getEmptyAffineMap();
  build(builder, state, srcMemref, dstMemref, srcMap, dstMap, srcIndices, dstIndices);
}

void CopyOp::build(OpBuilder &builder, OperationState &state, 
                    Value srcMemref, Value dstMemref, 
                    AffineMap srcMap, AffineMap dstMap,
                    ValueRange srcIndices, ValueRange dstIndices) {
  assert(srcMap.getNumInputs() == srcIndices.size() && 
    "source map inputs must match source indices count");
  assert(dstMap.getNumInputs() == dstIndices.size() && 
    "destination map inputs must match destination indices count");

  auto srcMemrefType = llvm::cast<MemRefType>(srcMemref.getType());
  auto dstMemrefType = llvm::cast<MemRefType>(dstMemref.getType());
  int64_t srcRank = srcMemrefType.getRank();
  int64_t dstRank = dstMemrefType.getRank();
  std::vector<int64_t> srcExtents;
  auto dstExtents = dstMemrefType.getShape(); 

  assert(srcRank >= dstRank && "src rank msut >= dst rank");
  for (unsigned i=0; i<srcRank; i++) {
    if (i < srcRank - dstRank) {
      srcExtents.push_back(1);
    } else {
      srcExtents.push_back(dstExtents[i-(srcRank-dstRank)]);
    }
  }

  build(builder, state, srcMemref, dstMemref, srcMap, dstMap, srcIndices, dstIndices, 
    builder.getDenseI64ArrayAttr(srcExtents), builder.getDenseI64ArrayAttr(dstExtents));
}

ParseResult CopyOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();
  // 解析源操作数和索引
  OpAsmParser::UnresolvedOperand srcMemrefInfo;
  AffineMapAttr srcMapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcMapOperands;
  if (parser.parseOperand(srcMemrefInfo) || parser.parseLSquare() ||
      parser.parseAffineMapOfSSAIds(srcMapOperands, srcMapAttr, "srcMap", result.attributes) ||
      parser.parseRSquare() || parser.parseComma())
    return failure();
  // 解析目标操作数和索引
  OpAsmParser::UnresolvedOperand dstMemrefInfo;
  AffineMapAttr dstMapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> dstMapOperands;
  if (parser.parseOperand(dstMemrefInfo) || parser.parseLSquare() ||
      parser.parseAffineMapOfSSAIds(dstMapOperands, dstMapAttr, "dstMap", result.attributes) ||
      parser.parseRSquare())
    return failure();
  // 解析类型信息
  SmallVector<Type, 2> memrefTypes;
  if (parser.parseColonTypeList(memrefTypes))
    return failure();
  if (memrefTypes.size() != 2)
    return parser.emitError(parser.getNameLoc(), "expected two memref types");
  auto srcType = dyn_cast<MemRefType>(memrefTypes[0]);
  auto dstType = dyn_cast<MemRefType>(memrefTypes[1]);
  if (!srcType || !dstType)
    return parser.emitError(parser.getNameLoc(), "expected memref types");
  // 解析可选的属性字典，但要排除 operandSegmentSizes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // 解析操作数
  if (parser.resolveOperand(srcMemrefInfo, srcType, result.operands) ||
      parser.resolveOperand(dstMemrefInfo, dstType, result.operands) ||
      parser.resolveOperands(srcMapOperands, indexTy, result.operands) ||
      parser.resolveOperands(dstMapOperands, indexTy, result.operands))
    return failure();
  // 设置 operandSegmentSizes 属性（隐藏的）
  SmallVector<int32_t> segmentSizes = {
      1, 1, 
      static_cast<int32_t>(srcMapOperands.size()),  // srcIndices
      static_cast<int32_t>(dstMapOperands.size())   // dstIndices
  };
  result.addAttribute(CopyOp::getOperandSegmentSizesAttrName(result.name), 
    builder.getDenseI32ArrayAttr(segmentSizes));
  return success();
}

void CopyOp::print(OpAsmPrinter &p) {
  p << " " << getSrcMemRef() << "[";
  // 打印源映射和操作数
  if (AffineMapAttr srcMapAttr = (*this)->getAttrOfType<AffineMapAttr>("srcMap")) {
    p.printAffineMapOfSSAIds(srcMapAttr, getSrcIndices());
  }
  p << "], " << getDstMemRef() << "[";
  // 打印目标映射和操作数
  if (AffineMapAttr dstMapAttr = (*this)->getAttrOfType<AffineMapAttr>("dstMap")) {
    p.printAffineMapOfSSAIds(dstMapAttr, getDstIndices());
  }
  auto srcExtentsAttr = getSrcExtentsAttr();
  auto dstExtentsAttr = getDstExtentsAttr();
  p << "] {src_extents = [";
  llvm::interleaveComma(srcExtentsAttr.asArrayRef(), p);
  p << "], dst_extents = [";
  llvm::interleaveComma(dstExtentsAttr.asArrayRef(), p);
  p << "]} ";
  
  // 打印属性字典，但要排除已打印的映射属性和 operandSegmentSizes
  SmallVector<StringRef> elidedAttrs = { "srcMap", "dstMap", "srcExtents", "dstExtents",
      CopyOp::getOperandSegmentSizesAttrName((*this)->getName()).getValue()
  };
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  p << " : " << getSrcMemRef().getType() << ", " << getDstMemRef().getType();
}

LogicalResult CopyOp::verify() {
  AffineMap srcMap = getSrcMap();
  AffineMap dstMap = getDstMap();
  // 正确：srcIndices 长度必须等于 srcMap 输入维度
  if (getSrcIndices().size() != srcMap.getNumInputs()) {
    return emitOpError("expected ") << srcMap.getNumInputs()
           << " source indices, but got " << getSrcIndices().size();
  }
  // 正确：dstIndices 长度必须等于 dstMap 输入维度
  if (getDstIndices().size() != dstMap.getNumInputs()) {
    return emitOpError("expected ") << dstMap.getNumInputs()
           << " destination indices, but got " << getDstIndices().size();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// -- FillOp --
//===----------------------------------------------------------------------===//
LogicalResult FillOp::verify() {
  auto memrefType = dyn_cast<MemRefType>(getMemref().getType());
  auto elemType = memrefType.getElementType();
  auto valueAttr = getValueAttr();
  if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    Type valueType = floatAttr.getType();
    if (elemType != valueType) {
      return emitOpError("value type ") << valueType << " does not match memref element type " << elemType;
    }
  } else {
    return emitOpError("fill value must be a float attribute");
  }
  return success();
}

// ParseResult FillOp::parse(OpAsmParser &parser, OperationState &result) {
//   OpAsmParser::UnresolvedOperand dstOperand;
//   Attribute valueAttr;
//   Type dstType;
//   if (parser.parseOperand(dstOperand) || parser.parseComma() ||
//       parser.parseAttribute(valueAttr, "value", result.attributes))// 解析目标操作数
//     return failure();
//   if (parser.parseColonType(dstType))// 解析类型
//     return failure();
//   if (parser.resolveOperand(dstOperand, dstType, result.operands))// 解析操作数
//     return failure();
//   return success();
// }

// void FillOp::print(OpAsmPrinter &p) {
//   p << " " << getMemref() << ", value = " << getValueAttr();
//   p << " : " << getMemref().getType();
// }

//===----------------------------------------------------------------------===//
// -- ReduceOp --
//===----------------------------------------------------------------------===//
void ReduceOp::build(OpBuilder &builder, OperationState &state, 
                    Value src, Value dst, StringRef kind, int64_t dim, bool clear) {
  state.addOperands({src, dst});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("dim", builder.getI64IntegerAttr(dim));
  state.addAttribute("clear", builder.getBoolAttr(clear));
}

LogicalResult ReduceOp::verify() {
  auto srcType = getSrcType();
  auto dstType = getDstType();
  auto kind = getKind();
  auto dim = getDim();
  if (!isa<MemRefType>(srcType) || !isa<MemRefType>(dstType)) {  // 检查源和目标是否是 memref 类型
    return emitOpError("source and destination must be memref types");
  }
  if (dim < 0 || dim >= srcType.getRank()) {  // 检查维度有效性
    return emitOpError("dimension ") << dim << " is out of range for source of rank " << srcType.getRank();
  }
  if (kind != "add" && kind != "mul" && kind != "min" && kind != "max") {  // 检查 reduce 类型有效性
    return emitOpError("unsupported reduce kind: ") << kind << ". Supported: add, mul, min, max";
  }
  // 检查源和目标形状兼容性
  auto srcShape = srcType.getShape();
  auto dstShape = dstType.getShape();
  if (srcType.getRank() != dstType.getRank() + 1) {  // 目标 rank 应该比源 rank 少 1
    return emitOpError("destination rank must be one less than source rank. Got: ")
           << dstType.getRank() << " vs " << srcType.getRank();
  }
  for (int64_t i = 0, j = 0; i < srcType.getRank(); ++i) {  // 检查非归约维度是否匹配
    if (i == dim) continue; // 跳过归约维度
    if (srcShape[i] != dstShape[j]) {
      return emitOpError("non-reduced dimension mismatch: source[")
             << i << "] = " << srcShape[i] << " vs destination["
             << j << "] = " << dstShape[j];
    }
    j++;
  }
  if (srcType.getElementType() != dstType.getElementType()) {  // 检查元素类型兼容性
    return emitOpError("source and destination must have the same element type");
  }
  return success();
}

// // 自定义汇编格式解析
// ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
//   OpAsmParser::UnresolvedOperand srcOperand, dstOperand;
//   Type srcType, dstType;
//   // 解析操作数: %src, %dst
//   if (parser.parseOperand(srcOperand) || parser.parseComma() || parser.parseOperand(dstOperand))
//     return failure();
//   if (parser.parseOptionalAttrDict(result.attributes))// 解析属性字典: {dim=0, clear=1, kind="max"}
//     return failure();
//   // 解析类型: : memref<1xf16>, memref<1024xf16>
//   if (parser.parseColon() || parser.parseType(srcType) || parser.parseComma() || parser.parseType(dstType))
//     return failure();
//   // 解析操作数
//   if (parser.resolveOperand(srcOperand, srcType, result.operands) || 
//       parser.resolveOperand(dstOperand, dstType, result.operands))
//     return failure();
//   return success();
// }

// // 自定义汇编格式打印
// void ReduceOp::print(OpAsmPrinter &p) {
//   p << " " << getSrc() << ", " << getDst();
//   // 打印属性字典
//   p << " {";
//   p << "dim = " << getDim();
//   p << ", clear = " << getClear();
//   p << ", kind = \"" << getKind() << "\"";
//   p << "}";
//   p << " : " << getSrc().getType() << ", " << getDst().getType();
// }



} // namespace frisk
} // namespace mlir