#include <vector>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "Dialect/Frisk/IR/FriskDialect.h"
#define GET_OP_CLASSES
#include "Dialect/Frisk/IR/FriskOps.cpp.inc"

// move dialect def in this file to make compiler happy
#include "Dialect/Frisk/IR/FriskDialect.cpp.inc"
namespace mlir {
namespace frisk {

void FriskDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Frisk/IR/FriskOps.cpp.inc"
      >();
}

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
  auto funcTy = mlir::dyn_cast<FunctionType>(funcType);
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

LogicalResult BlockOp::verify() {
  auto &blockBody = getRegion();
  if (blockBody.empty()) {
    return emitOpError("block must have a body");
  }
  auto &block = blockBody.front();
  bool hasAffineLoadOp = false;
  bool hasAffineStoreOp = false;
  
  for (auto &op : block.getOperations()) {
    if (isa<mlir::affine::AffineLoadOp>(op)) {    // 检查是否是affine.load操作
      hasAffineLoadOp = true;
    }
    else if (isa<mlir::affine::AffineStoreOp>(op)) {    // 检查是否是affine.store操作  
      hasAffineStoreOp = true;
    }
    // 如果已经找到两种操作，可以提前退出
    if (hasAffineLoadOp && hasAffineStoreOp) {
      break;
    }
  }
  // 验证结果
  if (!hasAffineLoadOp) {
    return emitOpError("block must contain at least one affine.load operation");
  }
  if (!hasAffineStoreOp) {
    return emitOpError("block must contain at least one affine.store operation");
  }
  return mlir::success();
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
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren, true) 
      || parser.parseRegion(*region, regionArgs))
    return failure();
  result.addRegion(std::move(region));
  return success();
}

void ForOp::print(OpAsmPrinter &p) {
  p << " ";
  p << getInductionVar() << " = " << getLower();  // 打印循环变量 //打印下界
  p << " to " << getUpper(); // 打印上界
  p << " step = " << getStep();  // 打印步长
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// -- GemmOp --
//===----------------------------------------------------------------------===//
LogicalResult GemmOp::verify() {
  auto AType = dyn_cast<mlir::MemRefType>(getA().getType());
  auto BType = dyn_cast<mlir::MemRefType>(getB().getType());
  auto CType = dyn_cast<mlir::MemRefType>(getC().getType());
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

} // namespace frisk
} // namespace mlir