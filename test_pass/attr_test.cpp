#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"

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
    auto typedAttr = parsedAttr.dyn_cast_or_null<mlir::frisk::MemorySpaceAttr>();
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
    auto typedAttr = parsedAttr.dyn_cast_or_null<mlir::frisk::GemmWarpPolicyAttr>();
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

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<mlir::frisk::FriskDialect>();
  MLIRContext context(registry);
  context.loadDialect<mlir::frisk::FriskDialect>();

  bool success = true;
  success &= testMemorySpaceAttr(context);
  success &= testGemmWarpPolicyAttr(context);

  llvm::outs() << "\n" << std::string(60, '=') << "\n";
  if (success) {
    llvm::outs() << "All Frisk attributes are usable.\n";
    return 0;
  }

  llvm::errs() << "Attribute tests failed.\n";
  return 1;
}
