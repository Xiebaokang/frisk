#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::frisk;

namespace {

struct BankStats {
  double avgMaxConflicts = 0.0;
  int64_t maxConflicts = 0;
  int64_t samples = 0;

  llvm::json::Object toJSON() const {
    llvm::json::Object obj;
    obj["avg_max_conflict"] = avgMaxConflicts;
    obj["max_conflict"] = maxConflicts;
    obj["samples"] = samples;
    return obj;
  }
};

attr::MemorySpace parseMemorySpace(llvm::StringRef value,
                                   attr::MemorySpace fallback) {
  if (value.empty())
    return fallback;
  if (value.equals_insensitive("shared"))
    return attr::MemorySpace::Shared;
  if (value.equals_insensitive("local"))
    return attr::MemorySpace::Local;
  if (value.equals_insensitive("global"))
    return attr::MemorySpace::Global;
  llvm::errs() << "Unknown memory space '" << value
               << "', falling back to default\n";
  return fallback;
}

std::string stringifyMemorySpaceStr(attr::MemorySpace space) {
  return attr::stringifyMemorySpace(space).str();
}

Type parseElementType(OpBuilder &builder, llvm::StringRef dtype) {
  if (dtype.equals_insensitive("fp16"))
    return builder.getF16Type();
  if (dtype.equals_insensitive("bf16"))
    return builder.getBF16Type();
  if (dtype.equals_insensitive("fp32"))
    return builder.getF32Type();
  if (dtype.equals_insensitive("fp64"))
    return builder.getF64Type();
  llvm::errs() << "Unsupported dtype '" << dtype << "', defaulting to fp16\n";
  return builder.getF16Type();
}

int64_t evaluateAffineExpr(AffineExpr expr,
                           llvm::ArrayRef<int64_t> dimValues) {
  switch (expr.getKind()) {
  case mlir::AffineExprKind::Constant:
    return llvm::cast<AffineConstantExpr>(expr).getValue();
  case mlir::AffineExprKind::DimId: {
    auto dim = llvm::cast<AffineDimExpr>(expr);
    unsigned pos = dim.getPosition();
    if (pos >= dimValues.size())
      return 0;
    return dimValues[pos];
  }
  case mlir::AffineExprKind::Add: {
    auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
    return evaluateAffineExpr(bin.getLHS(), dimValues) +
           evaluateAffineExpr(bin.getRHS(), dimValues);
  }
  case mlir::AffineExprKind::Mul: {
    auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
    return evaluateAffineExpr(bin.getLHS(), dimValues) *
           evaluateAffineExpr(bin.getRHS(), dimValues);
  }
  case mlir::AffineExprKind::FloorDiv: {
    auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
    int64_t numerator = evaluateAffineExpr(bin.getLHS(), dimValues);
    int64_t denominator = evaluateAffineExpr(bin.getRHS(), dimValues);
    if (denominator == 0)
      return 0;
    return numerator / denominator;
  }
  case mlir::AffineExprKind::CeilDiv: {
    auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
    int64_t numerator = evaluateAffineExpr(bin.getLHS(), dimValues);
    int64_t denominator = evaluateAffineExpr(bin.getRHS(), dimValues);
    if (denominator == 0)
      return 0;
    return llvm::divideCeil(numerator, denominator);
  }
  case mlir::AffineExprKind::Mod: {
    auto bin = llvm::cast<AffineBinaryOpExpr>(expr);
    int64_t value = evaluateAffineExpr(bin.getLHS(), dimValues);
    int64_t divisor = evaluateAffineExpr(bin.getRHS(), dimValues);
    if (divisor == 0)
      return 0;
    int64_t result = value % divisor;
    if (result < 0)
      result += std::abs(divisor);
    return result;
  }
  case mlir::AffineExprKind::SymbolId:
    // No symbols are expected in generated layouts.
    return 0;
  }
  return 0;
}

std::optional<BankStats> analyzeLayout(LayoutAttr layout, MemRefType type,
                                       unsigned laneVector,
                                       unsigned bankCount = 32,
                                       unsigned bankWidthBytes = 4,
                                       unsigned warpSize = 32) {
  if (!layout || !type)
    return std::nullopt;
  auto indexAttr = layout.getForwardIndex();
  if (!indexAttr)
    return std::nullopt;
  auto map = indexAttr.getValue();
  if (!map || map.getNumDims() < 2)
    return std::nullopt;

  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() < 2)
    return std::nullopt;
  int64_t rows = shape[shape.size() - 2];
  int64_t cols = shape[shape.size() - 1];
  if (rows <= 0 || cols <= 0)
    return std::nullopt;

  laneVector = std::max<unsigned>(1, laneVector);
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits % 8 != 0)
    return std::nullopt;
  int64_t elementBytes = elementBits / 8;
  if (elementBytes == 0)
    return std::nullopt;

  int64_t windowWidth =
      std::max<int64_t>(1, static_cast<int64_t>(warpSize) * laneVector);
  BankStats stats;
  AffineExpr expr = map.getResult(map.getNumResults() - 1);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t chunk = 0; chunk < cols; chunk += windowWidth) {
      SmallVector<int64_t> histogram(bankCount, 0);
      bool anyLane = false;
      for (int64_t lane = 0; lane < static_cast<int64_t>(warpSize); ++lane) {
        int64_t baseCol = chunk + lane * laneVector;
        for (unsigned v = 0; v < laneVector; ++v) {
          int64_t col = baseCol + static_cast<int64_t>(v);
          if (col >= cols)
            break;
          anyLane = true;
          llvm::SmallVector<int64_t> dims(map.getNumDims(), 0);
          // Assume the logical matrix dims occupy the last indices.
          dims[dims.size() - 2] = row;
          dims[dims.size() - 1] = col;
          int64_t index = evaluateAffineExpr(expr, dims);
          int64_t byteAddress = index * elementBytes;
          int64_t bank =
              ((byteAddress / static_cast<int64_t>(bankWidthBytes)) %
               static_cast<int64_t>(bankCount));
          if (bank < 0)
            bank += bankCount;
          histogram[bank]++;
        }
      }
      if (!anyLane)
        continue;
      stats.samples++;
      int64_t localMax = 0;
      for (int64_t count : histogram)
        localMax = std::max(localMax, count);
      stats.maxConflicts = std::max(stats.maxConflicts, localMax);
      stats.avgMaxConflicts += static_cast<double>(localMax);
    }
  }
  if (stats.samples > 0)
    stats.avgMaxConflicts /= static_cast<double>(stats.samples);
  return stats;
}

std::optional<BankStats> analyzeRowMajorBaseline(MemRefType type,
                                                 unsigned laneVector,
                                                 unsigned bankCount = 32,
                                                 unsigned bankWidthBytes = 4,
                                                 unsigned warpSize = 32) {
  if (!type)
    return std::nullopt;
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() < 2)
    return std::nullopt;
  int64_t rows = shape[shape.size() - 2];
  int64_t cols = shape[shape.size() - 1];
  if (rows <= 0 || cols <= 0)
    return std::nullopt;
  laneVector = std::max<unsigned>(1, laneVector);
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits % 8 != 0)
    return std::nullopt;
  int64_t elementBytes = elementBits / 8;
  int64_t windowWidth =
      std::max<int64_t>(1, static_cast<int64_t>(warpSize) * laneVector);
  BankStats stats;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t chunk = 0; chunk < cols; chunk += windowWidth) {
      SmallVector<int64_t> histogram(bankCount, 0);
      bool anyLane = false;
      for (int64_t lane = 0; lane < static_cast<int64_t>(warpSize); ++lane) {
        int64_t baseCol = chunk + lane * laneVector;
        for (unsigned v = 0; v < laneVector; ++v) {
          int64_t col = baseCol + static_cast<int64_t>(v);
          if (col >= cols)
            break;
          anyLane = true;
          int64_t index = row * cols + col;
          int64_t byteAddress = index * elementBytes;
          int64_t bank =
              ((byteAddress / static_cast<int64_t>(bankWidthBytes)) %
               static_cast<int64_t>(bankCount));
          if (bank < 0)
            bank += bankCount;
          histogram[bank]++;
        }
      }
      if (!anyLane)
        continue;
      stats.samples++;
      int64_t localMax = 0;
      for (int64_t count : histogram)
        localMax = std::max(localMax, count);
      stats.maxConflicts = std::max(stats.maxConflicts, localMax);
      stats.avgMaxConflicts += static_cast<double>(localMax);
    }
  }
  if (stats.samples > 0)
    stats.avgMaxConflicts /= static_cast<double>(stats.samples);
  return stats;
}

llvm::json::Value toJSON(ArrayRef<int64_t> values) {
  llvm::json::Array array;
  for (int64_t value : values)
    array.push_back(value);
  return llvm::json::Value(std::move(array));
}

llvm::json::Object describeLayout(StringRef label, Value value,
                                  LayoutAttr layout, MemRefType type,
                                  bool includeBankStats, unsigned laneVector) {
  llvm::json::Object obj;
  obj["operand"] = label.str();
  if (type) {
    obj["shape"] = toJSON(type.getShape()); 
    obj["element_bits"] = type.getElementTypeBitWidth();
    auto memSpace = attr::MemorySpace::Global;
    if (auto spaceAttr = type.getMemorySpace())
      if (auto spaceInt = dyn_cast<IntegerAttr>(spaceAttr))
        if (auto symbolic =
                attr::symbolizeMemorySpace(spaceInt.getInt()))
          memSpace = *symbolic;
    obj["memory_space"] = stringifyMemorySpaceStr(memSpace);
  }
  if (layout) {
    std::string indexStr;
    if (auto indexAttr = layout.getForwardIndex()) {
      llvm::raw_string_ostream os(indexStr);
      indexAttr.print(os);
      os.flush();
    }
    if (!indexStr.empty())
      obj["forward_index"] = indexStr;

    std::string threadStr;
    if (auto threadAttr = layout.getForwardThread()) {
      llvm::raw_string_ostream os(threadStr);
      threadAttr.print(os);
      os.flush();
    }
    if (!threadStr.empty())
      obj["forward_thread"] = threadStr;

    if (auto replicate = layout.getReplicateSize())
      obj["replicate"] = replicate.getInt();

    if (includeBankStats && type) {
      bool isShared = false;
      if (auto spaceAttr = type.getMemorySpace())
        if (auto spaceInt = dyn_cast<IntegerAttr>(spaceAttr))
          if (auto symbolic =
                  attr::symbolizeMemorySpace(spaceInt.getInt()))
            isShared = (*symbolic == attr::MemorySpace::Shared);
      if (isShared) {
        if (auto stats = analyzeLayout(layout, type, laneVector))
          obj["bank_stats"] = stats->toJSON();
        if (auto baseline = analyzeRowMajorBaseline(type, laneVector))
          obj["baseline_row_major"] = baseline->toJSON();
      }
    }
  }
  return obj;
}

} // namespace

static llvm::cl::opt<std::string>
    targetOpt("target", llvm::cl::desc("Target triple (e.g. sm_80)"),
              llvm::cl::init("sm_80"));
static llvm::cl::opt<int>
    blockMOpt("block-m", llvm::cl::desc("GEMM tile M dimension"),
              llvm::cl::init(128));
static llvm::cl::opt<int>
    blockNOpt("block-n", llvm::cl::desc("GEMM tile N dimension"),
              llvm::cl::init(128));
static llvm::cl::opt<int>
    blockKOpt("block-k", llvm::cl::desc("GEMM tile K dimension"),
              llvm::cl::init(64));
static llvm::cl::opt<std::string>
    dtypeOpt("dtype", llvm::cl::desc("Element dtype (fp16, bf16, fp32)"),
             llvm::cl::init("fp16"));
static llvm::cl::opt<std::string>
    aSpaceOpt("a-space", llvm::cl::desc("Memory space for operand A"),
              llvm::cl::init("shared"));
static llvm::cl::opt<std::string>
    bSpaceOpt("b-space", llvm::cl::desc("Memory space for operand B"),
              llvm::cl::init("shared"));
static llvm::cl::opt<std::string>
    cSpaceOpt("c-space", llvm::cl::desc("Memory space for operand C"),
              llvm::cl::init("local"));
static llvm::cl::opt<int>
    threadsOpt("threads", llvm::cl::desc("Threads per block"), llvm::cl::init(128));
static llvm::cl::list<int>
    gridOpt("grid", llvm::cl::desc("Parallel grid dimensions (comma separated)"),
            llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);
static llvm::cl::opt<bool>
    transAOpt("trans-a", llvm::cl::desc("Use transposed operand A"),
              llvm::cl::init(false));
static llvm::cl::opt<bool>
    transBOpt("trans-b", llvm::cl::desc("Use transposed operand B"),
              llvm::cl::init(false));
static llvm::cl::opt<unsigned>
    laneVectorOpt("lane-vector",
                  llvm::cl::desc("Elements loaded per lane (default: 1)"),
                  llvm::cl::init(1));
static llvm::cl::opt<bool>
    ldmatrixOpt("ldmatrix",
                llvm::cl::desc("Approximate ldmatrix.x4 loads (forces lane-vector>=4)"),
                llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::InitLLVM initLL(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Frisk GEMM layout analysis tool\n");

  DialectRegistry registry;
  registry.insert<frisk::FriskDialect, affine::AffineDialect,
                  func::FuncDialect, arith::ArithDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(loc);
  module->setAttr("frisk.target", builder.getStringAttr(targetOpt));
  builder.setInsertionPointToEnd(module.getBody());

  Type elementType = parseElementType(builder, dtypeOpt);
  auto makeMemRef = [&](int64_t rows, int64_t cols,
                        attr::MemorySpace space) -> MemRefType {
    return MemRefType::get({rows, cols}, elementType, {},
                           static_cast<unsigned>(space));
  };

  attr::MemorySpace aSpace = parseMemorySpace(aSpaceOpt, attr::MemorySpace::Shared);
  attr::MemorySpace bSpace = parseMemorySpace(bSpaceOpt, attr::MemorySpace::Shared);
  attr::MemorySpace cSpace = parseMemorySpace(cSpaceOpt, attr::MemorySpace::Local);

  MemRefType aType = makeMemRef(blockMOpt, blockKOpt, aSpace);
  MemRefType bType = makeMemRef(blockKOpt, blockNOpt, bSpace);
  MemRefType cType = makeMemRef(blockMOpt, blockNOpt, cSpace);

  unsigned laneVector = std::max<unsigned>(1, laneVectorOpt.getValue());
  if (ldmatrixOpt)
    laneVector = std::max<unsigned>(laneVector, 4);

  SmallVector<Type> kernelArgs = {aType, bType, cType};
  auto kernelType = builder.getFunctionType(kernelArgs, {});
  auto kernel = builder.create<KernelOp>(loc, "test_kernel", kernelType);
  Block *entry = kernel.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  SmallVector<int64_t> gridShape;
  if (!gridOpt.empty())
    gridShape.assign(gridOpt.begin(), gridOpt.end());
  if (gridShape.empty())
    gridShape = {1, 1};

  auto parallel = builder.create<ParallelOp>(loc, gridShape, threadsOpt);
  Block *parallelBody = parallel.addEntryBlock();
  builder.setInsertionPointToStart(parallelBody);

  Value aVal = entry->getArgument(0);
  Value bVal = entry->getArgument(1);
  Value cVal = entry->getArgument(2);

  auto policyAttr = GemmWarpPolicyAttr::get(&context,
                                                  attr::GemmWarpPolicy::Square);
  auto boolAttr = [&](bool value) { return builder.getBoolAttr(value); };

  builder.create<GemmOp>(
      loc, aVal, bVal, cVal, boolAttr(transAOpt), boolAttr(transBOpt),
      builder.getI64IntegerAttr(blockMOpt),
      builder.getI64IntegerAttr(blockNOpt),
      builder.getI64IntegerAttr(blockKOpt), policyAttr, boolAttr(false));
  builder.create<EndOp>(loc);

  builder.setInsertionPointAfter(parallel);
  builder.create<EndOp>(loc);

  DenseMap<Value, Attribute> layoutMap;
  OpBuilder inferBuilder(&context);
  inferBuilder.setInsertionPoint(parallel.getOperation());
  if (failed(parallel.inferLayout(inferBuilder, layoutMap))) {
    llvm::errs() << "Layout inference failed\n";
    return 1;
  }

  llvm::json::Object result;
  {
    llvm::json::Object config;
    config["target"] = targetOpt;
    config["dtype"] = dtypeOpt;
    config["threads"] = static_cast<int64_t>(threadsOpt.getValue());
    llvm::json::Object block;
    block["M"] = static_cast<int64_t>(blockMOpt.getValue());
    block["N"] = static_cast<int64_t>(blockNOpt.getValue());
    block["K"] = static_cast<int64_t>(blockKOpt.getValue());
    config["block_shape"] = std::move(block);
    llvm::json::Array gridJson;
    for (int64_t dim : gridShape)
      gridJson.push_back(dim);
    config["grid"] = std::move(gridJson);
    llvm::json::Object spaces;
    spaces["A"] = stringifyMemorySpaceStr(aSpace);
    spaces["B"] = stringifyMemorySpaceStr(bSpace);
    spaces["C"] = stringifyMemorySpaceStr(cSpace);
    config["memory_spaces"] = std::move(spaces);
    config["lane_vector"] = static_cast<int64_t>(laneVector);
    config["ldmatrix_like"] = static_cast<bool>(ldmatrixOpt.getValue());
    result["config"] = std::move(config);
  }

  llvm::json::Array layouts;
  auto getLayoutFor = [&](Value value) -> LayoutAttr {
    auto it = layoutMap.find(value);
    if (it == layoutMap.end())
      return LayoutAttr();
    return dyn_cast<LayoutAttr>(it->second);
  };

  layouts.push_back(describeLayout("A", aVal, getLayoutFor(aVal), aType,
                                   /*includeBankStats=*/true, laneVector));
  layouts.push_back(describeLayout("B", bVal, getLayoutFor(bVal), bType,
                                   /*includeBankStats=*/true, laneVector));
  layouts.push_back(describeLayout("C", cVal, getLayoutFor(cVal), cType,
                                   /*includeBankStats=*/false, laneVector));

  result["layouts"] = std::move(layouts);

  llvm::json::OStream jsonOS(llvm::outs(), 2);
  jsonOS.value(llvm::json::Value(std::move(result)));
  llvm::outs() << "\n";
  return 0;
}
