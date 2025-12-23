#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace {

void printHeader(StringRef title) {
  llvm::outs() << "\n" << std::string(60, '=') << "\n";
  llvm::outs() << "-- " << title << " --\n";
  llvm::outs() << std::string(60, '-') << "\n";
}

void dumpAttribute(Attribute attr) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  attr.print(os);
  os.flush();
  llvm::outs() << buffer;
}

bool testMemorySpaceAttr(MLIRContext &context) {
  printHeader("MemorySpaceAttr");

  bool ok = true;
  for (auto space : {mlir::frisk::attr::MemorySpace::Local,
                     mlir::frisk::attr::MemorySpace::Global,
                     mlir::frisk::attr::MemorySpace::Shared}) {
    auto attr = mlir::frisk::MemorySpaceAttr::get(&context, space);
    if (!attr) {
      llvm::errs() << "Failed to construct MemorySpaceAttr for value "
                   << static_cast<int>(space) << "\n";
      ok = false;
      continue;
    }

    llvm::outs() << "Constructed: ";
    dumpAttribute(attr);
    llvm::outs() << " (enum = " << static_cast<unsigned>(attr.getValue())
                 << ")\n";
  }

  const char *moduleAsm =
      R"mlir(module attributes {frisk.test_attr = #frisk<memory_space Shared>} {})mlir";
  auto module = parseSourceString<ModuleOp>(moduleAsm, &context);
  if (!module) {
    llvm::errs() << "Failed to parse MemorySpaceAttr from assembly\n";
    ok = false;
  } else {
    Attribute parsedAttr = module->getOperation()->getAttr("frisk.test_attr");
    auto typedAttr = dyn_cast<mlir::frisk::MemorySpaceAttr>(parsedAttr);
    if (!typedAttr) {
      llvm::errs() << "Attribute parsed, but type mismatch\n";
      ok = false;
    } else {
      llvm::outs() << "Parsed: ";
      dumpAttribute(typedAttr);
      llvm::outs() << " -> enum = "
                   << static_cast<unsigned>(typedAttr.getValue()) << "\n";
    }
  }

  return ok;
}

bool testGemmWarpPolicyAttr(MLIRContext &context) {
  printHeader("GemmWarpPolicyAttr");

  bool ok = true;
  for (auto policy : {mlir::frisk::attr::GemmWarpPolicy::Square,
                      mlir::frisk::attr::GemmWarpPolicy::FullRow,
                      mlir::frisk::attr::GemmWarpPolicy::FullCol}) {
    auto attr = mlir::frisk::GemmWarpPolicyAttr::get(&context, policy);
    if (!attr) {
      llvm::errs() << "Failed to construct GemmWarpPolicyAttr for value "
                   << static_cast<int>(policy) << "\n";
      ok = false;
      continue;
    }

    llvm::outs() << "Constructed: ";
    dumpAttribute(attr);
    llvm::outs() << " (enum = " << static_cast<unsigned>(attr.getValue())
                 << ")\n";
  }

  const char *moduleAsm =
      R"mlir(module attributes {frisk.test_attr = #frisk<gemm_warp_policy FullRow>} {})mlir";
  auto module = parseSourceString<ModuleOp>(moduleAsm, &context);
  if (!module) {
    llvm::errs() << "Failed to parse GemmWarpPolicyAttr from assembly\n";
    ok = false;
  } else {
    Attribute parsedAttr = module->getOperation()->getAttr("frisk.test_attr");
    auto typedAttr = dyn_cast<mlir::frisk::GemmWarpPolicyAttr>(parsedAttr);
    if (!typedAttr) {
      llvm::errs() << "Attribute parsed, but type mismatch\n";
      ok = false;
    } else {
      llvm::outs() << "Parsed: ";
      dumpAttribute(typedAttr);
      llvm::outs() << " -> enum = "
                   << static_cast<unsigned>(typedAttr.getValue()) << "\n";
    }
  }

  return ok;
}

bool testLayoutAttr(MLIRContext &context) {
  printHeader("LayoutAttr");
  OpBuilder builder(&context);
  DenseI64ArrayAttr shape = builder.getDenseI64ArrayAttr({4, 4});
  AffineExpr i = builder.getAffineDimExpr(0);
  AffineExpr j = builder.getAffineDimExpr(1);
  AffineExpr linear = i * builder.getAffineConstantExpr(4) + j;
  auto indexMap = AffineMapAttr::get(AffineMap::get(2, 0, linear));
  auto layout = mlir::frisk::LayoutAttr::get(&context, shape, indexMap,
                                             AffineMapAttr(), IntegerAttr());
  if (!layout) {
    llvm::errs() << "Failed to construct LayoutAttr\n";
    return false;
  }
  llvm::outs() << "Constructed layout: ";
  dumpAttribute(layout);
  llvm::outs() << "\n";
  return true;
}

static const char *describeMemorySpace(mlir::frisk::attr::MemorySpace space) {
  switch (space) {
  case mlir::frisk::attr::MemorySpace::Local:
    return "local";
  case mlir::frisk::attr::MemorySpace::Global:
    return "global";
  case mlir::frisk::attr::MemorySpace::Shared:
    return "shared";
  }
  return "unknown";
}

static bool runGemmLayoutCase(MLIRContext &context, StringRef caseLabel,
                              StringRef target,
                              mlir::frisk::attr::MemorySpace spaceA,
                              mlir::frisk::attr::MemorySpace spaceB) {
  OpBuilder moduleBuilder(&context);
  auto loc = moduleBuilder.getUnknownLoc();
  OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
  module->getOperation()->setAttr("frisk.target",
                                  moduleBuilder.getStringAttr(target));

  auto f16 = moduleBuilder.getF16Type();
  auto makeSpaceAttr = [&](mlir::frisk::attr::MemorySpace space) {
    return moduleBuilder.getI64IntegerAttr(static_cast<int64_t>(space));
  };
  auto memAType =
      MemRefType::get({128, 64}, f16, AffineMapAttr(), makeSpaceAttr(spaceA));
  auto memBType =
      MemRefType::get({64, 128}, f16, AffineMapAttr(), makeSpaceAttr(spaceB));
  auto memCType = MemRefType::get({128, 128}, f16, AffineMapAttr(),
                                  makeSpaceAttr(mlir::frisk::attr::MemorySpace::Local));

  auto funcType =
      moduleBuilder.getFunctionType({memAType, memBType, memCType}, {});
  Block &moduleBlock = module->getBodyRegion().front();
  moduleBuilder.setInsertionPointToStart(&moduleBlock);
  auto kernel =
      moduleBuilder.create<mlir::frisk::KernelOp>(loc, caseLabel, funcType);
  Block *entry = kernel.addEntryBlock();

  OpBuilder bodyBuilder(&context);
  bodyBuilder.setInsertionPoint(entry->getTerminator());
  auto gemm = bodyBuilder.create<mlir::frisk::GemmOp>(
      loc, entry->getArgument(0), entry->getArgument(1),
      entry->getArgument(2), false, false, static_cast<uint64_t>(128),
      static_cast<uint64_t>(128), static_cast<uint64_t>(64),
      mlir::frisk::attr::GemmWarpPolicy::Square, false);
  gemm->setAttr("frisk.threads", bodyBuilder.getI64IntegerAttr(128));

  llvm::outs() << "\nCase '" << caseLabel << "' (target=" << target
               << ", A=" << describeMemorySpace(spaceA)
               << ", B=" << describeMemorySpace(spaceB) << ")\n";

  module->print(llvm::outs());

  OpBuilder builder(&context);
  builder.setInsertionPoint(gemm);
  llvm::DenseMap<Value, Attribute> layoutMap;
  if (failed(gemm.inferLayout(builder, layoutMap))) {
    llvm::errs() << "inferLayout returned failure for case '" << caseLabel
                 << "'\n";
    return false;
  }

  auto checkOperand = [&](Value value, StringRef label) -> bool {
    auto it = layoutMap.find(value);
    if (it == layoutMap.end()) {
      llvm::errs() << "Missing layout for operand " << label << "";
      return false;
    }
    auto layout = dyn_cast<mlir::frisk::LayoutAttr>(it->second);
    if (!layout) {
      llvm::errs() << "Layout for operand " << label << " has wrong type";
      return false;
    }
    llvm::outs() << label << " layout: "
                 << mlir::frisk::layoutDebugString(layout) << "\n";
    return true;
  };
  bool ok = true;
  ok &= checkOperand(gemm.getA(), "A");
  ok &= checkOperand(gemm.getB(), "B");
  ok &= checkOperand(gemm.getC(), "C");
  return ok;
}

bool testGemmLayoutInference(MLIRContext &context) {
  printHeader("Gemm Layout Inference");
  bool ok = true;
  ok &= runGemmLayoutCase(context, "sm80_ss", "sm_80",
                          mlir::frisk::attr::MemorySpace::Shared,
                          mlir::frisk::attr::MemorySpace::Shared);
  llvm::outs() << std::string(60, '-') << "\n";

  ok &= runGemmLayoutCase(context, "sm80_rs", "sm_80",
    mlir::frisk::attr::MemorySpace::Local,
    mlir::frisk::attr::MemorySpace::Shared);
  llvm::outs() << std::string(60, '-') << "\n";

  ok &= runGemmLayoutCase(context, "sm80_sr", "sm_80",
    mlir::frisk::attr::MemorySpace::Shared,
    mlir::frisk::attr::MemorySpace::Local);
  llvm::outs() << std::string(60, '-') << "\n";

  ok &= runGemmLayoutCase(context, "sm90_ss", "sm_90",
                          mlir::frisk::attr::MemorySpace::Shared,
                          mlir::frisk::attr::MemorySpace::Shared);
  llvm::outs() << std::string(60, '-') << "\n";

  ok &= runGemmLayoutCase(context, "sm90_rs", "sm_90",
    mlir::frisk::attr::MemorySpace::Local,
    mlir::frisk::attr::MemorySpace::Shared);
  return ok;
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<mlir::frisk::FriskDialect>();
  MLIRContext context(registry);
  context.loadDialect<mlir::frisk::FriskDialect>();

  bool success = true;
  success &= testMemorySpaceAttr(context);
  success &= testGemmWarpPolicyAttr(context);
  success &= testLayoutAttr(context);
  success &= testGemmLayoutInference(context);

  llvm::outs() << "\n" << std::string(60, '=') << "\n";
  if (success) {
    llvm::outs() << "All Frisk attributes are usable.\n";
    return 0;
  }

  llvm::errs() << "Attribute tests failed.\n";
  return 1;
}
