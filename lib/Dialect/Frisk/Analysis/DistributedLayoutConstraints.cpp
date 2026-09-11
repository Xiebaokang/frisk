#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::frisk {
LogicalResult collectDistributedLayoutConstraints(
    Operation *root, LayoutConstraintGraph &graph, LayoutConstraintBuilder &builder) {
  DenseMap<Value, bool> visited;
  DenseMap<Operation *, SmallVector<LayoutVarID>> functionResults;
  auto bind = [&](Value value) -> LogicalResult {
    if (!isa<TensorType>(value.getType()) || !visited.try_emplace(value, true).second)
      return success();
    Operation *anchor = value.getDefiningOp();
    if (!anchor)
      anchor = cast<BlockArgument>(value).getOwner()->getParentOp();
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      return anchor->emitError("distributed inference requires ranked tensor types");
    if (type.getRank() == 0)
      return anchor->emitError("distributed inference requires nonzero-rank tensor tiles");
    if (!type.hasStaticShape() || llvm::any_of(type.getShape(), [](int64_t extent) {
          return extent <= 1 || !llvm::isPowerOf2_64(extent);
        }))
      return anchor->emitError("distributed inference requires static power-of-two tile extents greater than one (M1 bit-width boundary)");
    LayoutVarID id = builder.getOrCreateDistributedVar(value);
    if (Attribute encoding = type.getEncoding()) {
      auto distributed = dyn_cast<DistributedEncodingAttr>(encoding);
      if (!distributed || failed(distributed.verifyForType(type, anchor->getLoc())))
        return anchor->emitError("unsupported tensor layout encoding");
      return builder.require(id, encoding, anchor, "tensor-encoding");
    }
    return success();
  };
  auto connectUse = [&](OpOperand &use, LayoutVarID expected) {
    return builder.convertible(builder.getOrCreateDistributedVar(use.get()),
                                expected, use);
  };
  auto result = root->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    bool tensorBearing = llvm::any_of(op->getOperandTypes(), [](Type t) {
      return isa<TensorType>(t);
    }) || llvm::any_of(op->getResultTypes(), [](Type t) { return isa<TensorType>(t); });
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          tensorBearing |= isa<TensorType>(arg.getType());
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      tensorBearing |= llvm::any_of(function.getResultTypes(), [](Type t) {
        return isa<TensorType>(t);
      });
      tensorBearing |= llvm::any_of(function.getArgumentTypes(), [](Type t) {
        return isa<TensorType>(t);
      });
    }
    if (!tensorBearing)
      return WalkResult::advance();
    for (Value operand : op->getOperands())
      if (failed(bind(operand))) return WalkResult::interrupt();
    for (Value value : op->getResults())
      if (failed(bind(value))) return WalkResult::interrupt();
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (failed(bind(arg))) return WalkResult::interrupt();

    if (auto function = dyn_cast<func::FuncOp>(op)) {
      if (function.isExternal()) {
        op->emitError("operation has no layout constraint model for external tensor signature");
        return WalkResult::interrupt();
      }
      SmallVector<LayoutVarID> ids;
      for (auto [index, type] : llvm::enumerate(function.getResultTypes())) {
        if (!isa<RankedTensorType>(type)) {
          ids.push_back(std::numeric_limits<LayoutVarID>::max());
          continue;
        }
        // Function symbol paths are stable and unique within the module tree.
        std::string name;
        for (Operation *owner = op; owner; owner = owner->getParentOp())
          if (auto symbol = owner->getAttrOfType<StringAttr>("sym_name"))
            name = std::to_string(symbol.getValue().size()) + ":" + symbol.getValue().str() + "/" + name;
        auto id = graph.addVariable(LayoutKind::Distributed, type,
                                    name + "function-result" + std::to_string(index), op);
        graph.getVariable(id).functionResult = index;
        if (auto encoding = cast<RankedTensorType>(type).getEncoding())
          if (failed(builder.require(id, encoding, op, "function-result-encoding")))
            return WalkResult::interrupt();
        ids.push_back(id);
      }
      functionResults[op] = std::move(ids);
      return WalkResult::advance();
    }
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      auto function = op->getParentOfType<func::FuncOp>();
      auto it = functionResults.find(function);
      if (it == functionResults.end()) {
        op->emitError("missing tensor function result layout model");
        return WalkResult::interrupt();
      }
      for (OpOperand &use : op->getOpOperands())
        if (isa<RankedTensorType>(use.get().getType()))
          (void)connectUse(use, it->second[use.getOperandNumber()]);
      return WalkResult::advance();
    }
    if (isa<scf::IfOp>(op)) {
      // Yield uses are connected below, directly to each result join slot.
      return WalkResult::advance();
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      for (auto [index, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
        if (!isa<RankedTensorType>(arg.getType())) continue;
        auto slot = builder.getOrCreateDistributedVar(arg);
        (void)builder.same(slot, builder.getOrCreateDistributedVar(loop.getResult(index)),
                           op, "for-carried-slot");
        (void)connectUse(loop.getInitsMutable()[index], slot);
      }
      return WalkResult::advance();
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      // The input/before/yield tuple and condition/after/result tuple can
      // have different arities and types. Do not tie unrelated tuple slots.
      for (auto [index, arg] : llvm::enumerate(loop.getBeforeArguments()))
        if (isa<RankedTensorType>(arg.getType()))
          (void)connectUse(op->getOpOperand(index),
                           builder.getOrCreateDistributedVar(arg));
      for (auto [index, arg] : llvm::enumerate(loop.getAfterArguments()))
        if (isa<RankedTensorType>(arg.getType()))
          (void)builder.same(builder.getOrCreateDistributedVar(arg),
              builder.getOrCreateDistributedVar(loop.getResult(index)), op,
              "while-result-slot");
      return WalkResult::advance();
    }
    if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      Operation *parent = op->getParentOp();
      for (OpOperand &use : op->getOpOperands()) {
        if (!isa<RankedTensorType>(use.get().getType())) continue;
        unsigned index = use.getOperandNumber();
        Value expected;
        if (isa<scf::IfOp, scf::ForOp>(parent))
          expected = parent->getResult(index);
        else if (auto loop = dyn_cast<scf::WhileOp>(parent))
          expected = loop.getBeforeArguments()[index];
        else {
          op->emitError("operation has no layout constraint model for tensor yield");
          return WalkResult::interrupt();
        }
        (void)connectUse(use, builder.getOrCreateDistributedVar(expected));
      }
      return WalkResult::advance();
    }
    if (auto condition = dyn_cast<scf::ConditionOp>(op)) {
      auto loop = cast<scf::WhileOp>(op->getParentOp());
      for (OpOperand &use : op->getOpOperands().drop_front())
        if (isa<RankedTensorType>(use.get().getType()))
          (void)connectUse(use, builder.getOrCreateDistributedVar(
              loop.getResult(use.getOperandNumber() - 1)));
      return WalkResult::advance();
    }
    if (auto load = dyn_cast<TileLoadOp>(op)) {
      if (!load.getSource().getDefiningOp<LayoutViewOp>()) {
        op->emitError("distributed tile access requires a layout_view source");
        return WalkResult::interrupt();
      }
      (void)builder.storageAccess(builder.getOrCreateDistributedVar(load.getResult()),
          builder.getOrCreateStorageVar(load.getSource()), AccessKind::Read, op, "tile-load");
      return WalkResult::advance();
    }
    if (auto store = dyn_cast<TileStoreOp>(op)) {
      if (!store.getTarget().getDefiningOp<LayoutViewOp>()) {
        op->emitError("distributed tile access requires a layout_view target");
        return WalkResult::interrupt();
      }
      (void)builder.storageAccess(builder.getOrCreateDistributedUse(op->getOpOperand(0)),
          builder.getOrCreateStorageVar(store.getTarget()), AccessKind::Write, op, "tile-store");
      return WalkResult::advance();
    }
    if (auto convert = dyn_cast<ConvertLayoutOp>(op)) {
      (void)builder.convertible(builder.getOrCreateDistributedVar(convert.getSource()),
          builder.getOrCreateDistributedVar(convert.getResult()), op->getOpOperand(0), true);
      return WalkResult::advance();
    }
    if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
      if (op->getNumResults() != 1 || !isa<RankedTensorType>(op->getResult(0).getType())) {
        op->emitError("operation has no layout constraint model for non-tensor transpose");
        return WalkResult::interrupt();
      }
      auto dst = builder.getOrCreateDistributedVar(op->getResult(0));
      (void)builder.transform(builder.getOrCreateDistributedUse(op->getOpOperand(0)),
          dst, transpose.getPermutationAttr(), op, "transpose");
      (void)connectUse(op->getOpOperand(1), dst);
      return WalkResult::advance();
    }
    if (isa<tensor::EmptyOp>(op))
      return WalkResult::advance();
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "arith" && op->getName().getStringRef() == "arith.constant") {
      if (!isa_and_nonnull<DenseElementsAttr>(op->getAttr("value"))) {
        op->emitError("operation has no layout constraint model for non-dense tensor constant");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if ((dialect == "arith" || dialect == "math") &&
        op->hasTrait<OpTrait::Elementwise>() && op->getNumResults() > 0 &&
        isa<RankedTensorType>(op->getResult(0).getType())) {
      auto expected = builder.getOrCreateDistributedVar(op->getResult(0));
      auto shape = cast<RankedTensorType>(op->getResult(0).getType()).getShape();
      for (Value value : op->getResults()) {
        auto type = dyn_cast<RankedTensorType>(value.getType());
        if (!type || type.getShape() != shape) {
          op->emitError("elementwise tensor layout model requires identical shapes");
          return WalkResult::interrupt();
        }
        if (value != op->getResult(0))
          (void)builder.same(expected, builder.getOrCreateDistributedVar(value), op, "elementwise");
      }
      for (OpOperand &use : op->getOpOperands()) {
        auto type = dyn_cast<RankedTensorType>(use.get().getType());
        if (!type) continue;
        if (type.getShape() != shape) {
          op->emitError("elementwise tensor layout model requires identical shapes");
          return WalkResult::interrupt();
        }
        // Converting an operand preserves its element type; the relation may
        // tie its layout to a result of a different element type (e.g. cmpf).
        if (type.getElementType() != cast<RankedTensorType>(op->getResult(0).getType()).getElementType()) {
          auto operand = builder.getOrCreateDistributedUse(use);
          (void)builder.same(operand, expected, op, "elementwise");
        } else {
          (void)connectUse(use, expected);
        }
      }
      return WalkResult::advance();
    }
    op->emitError("operation has no layout constraint model");
    return WalkResult::interrupt();
  });
  return success(!result.wasInterrupted());
}
} // namespace mlir::frisk
