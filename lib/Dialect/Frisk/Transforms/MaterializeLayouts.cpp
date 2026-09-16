#include "Dialect/Frisk/Analysis/LayoutVerifier.h"

#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Transforms/LayoutTypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir::frisk {

namespace {
struct ConversionSnapshot {
  unsigned operand;
  Attribute source;
  Attribute target;
};

/// Owns no original IR. Until commit, all lookups use still-live original
/// identities; every constructed op is owned by the detached module.
class LayoutRebuild {
public:
  LayoutRebuild(const LayoutConstraintGraph &graph, const LayoutSolution &solution)
      : graph(graph), solution(solution), converter(graph, solution) {}

  LogicalResult prepare(Operation *root, ArrayRef<LayoutConversionEdge> edges) {
    for (const auto &edge : edges) {
      Operation *owner = edge.use->getOwner();
      if (!root->isAncestor(owner))
        return root->emitError("conversion consumer is outside materialization root");
      unsigned operand = edge.use->getOperandNumber();
      auto &list = conversions[owner];
      if (llvm::any_of(list, [&](auto snapshot) { return snapshot.operand == operand; }))
        return owner->emitError("duplicate conversion consumer operand");
      list.push_back({operand, edge.sourceEncoding, edge.targetEncoding});
    }
    for (const LayoutVar &var : graph.getVariables()) {
      if (var.kind == LayoutKind::Storage) {
        auto view = dyn_cast_or_null<LayoutViewOp>(var.anchor);
        auto layout = dyn_cast_or_null<StorageLayoutAttr>(solution.assignments.lookup(var.id));
        if (!view || !layout || !root->isAncestor(view))
          return root->emitError("invalid storage materialization binding");
        if (view.getLayoutAttr() && view.getLayoutAttr() != layout)
          return view.emitError("materialization would overwrite an explicit hard binding");
        storage[var.anchor] = layout;
      } else if (var.value) {
        auto type = converter.convertLayoutBearingTensor(var.value);
        if (failed(type)) return failure();
        types[var.value] = *type;
      } else if (var.functionResult) {
        auto function = dyn_cast_or_null<func::FuncOp>(var.anchor);
        auto encoding = solution.assignments.lookup(var.id);
        if (!function || *var.functionResult >= function.getNumResults())
          return root->emitError("invalid function result layout binding");
        auto type = cast<RankedTensorType>(function.getResultTypes()[*var.functionResult]);
        if (type.getEncoding() && type.getEncoding() != encoding)
          return function.emitError("materialization would overwrite an explicit hard binding");
        functionResults[{var.anchor, *var.functionResult}] =
            RankedTensorType::get(type.getShape(), type.getElementType(), encoding);
      }
    }
    return success();
  }

  LogicalResult rebuildRegion(Region &original, Region &replacement) {
    // All block arguments/successors are available before visiting any op.
    // A worklist keeps original per-block order but also handles scalar CFGs
    // whose printed block order is not dominance order.
    SmallVector<std::pair<Block *, Block::iterator>> pending;
    for (Block &block : original) {
      auto *copy = new Block();
      replacement.push_back(copy);
      mapping.map(&block, copy);
      for (BlockArgument arg : block.getArguments()) {
        auto type = replacementType(arg);
        if (failed(type)) return failure();
        mapping.map(arg, copy->addArgument(*type, arg.getLoc()));
      }
      pending.emplace_back(&block, block.begin());
    }
    while (!pending.empty()) {
      bool progressed = false;
      for (auto &entry : pending) {
        Block *block = entry.first;
        auto &next = entry.second;
        while (next != block->end()) {
          Operation *op = &*next;
          if (llvm::any_of(op->getOperands(), [&](Value value) {
                return !mapping.contains(value);
              }))
            break;
          llvm::SetVector<Value> captures;
          getUsedValuesDefinedAbove(op->getRegions(), captures);
          if (llvm::any_of(captures, [&](Value value) {
                return !mapping.contains(value);
              }))
            break;
          if (failed(rebuildOperation(op, mapping.lookup(block))))
            return failure();
          ++next;
          progressed = true;
        }
      }
      llvm::erase_if(pending, [](const auto &entry) {
        return entry.second == entry.first->end();
      });
      if (!pending.empty() && !progressed)
        return pending.front().first->getParentOp()->emitError(
            "unmapped SSA dependency during layout rebuilding");
    }
    return success();
  }

private:
  FailureOr<Type> replacementType(Value value) {
    if (!isa<TensorType>(value.getType())) return value.getType();
    auto found = types.find(value);
    if (found == types.end()) {
      emitError(value.getLoc()) << "tensor has no distributed definition binding";
      return failure();
    }
    return found->second;
  }

  LogicalResult rebuildOperation(Operation *op, Block *block) {
    SmallVector<Value> operands;
    for (Value value : op->getOperands()) operands.push_back(mapping.lookup(value));
    OpBuilder builder(block, block->end());
    for (const auto &edge : conversions.lookup(op)) {
      auto source = dyn_cast<RankedTensorType>(operands[edge.operand].getType());
      if (!source || source.getEncoding() != edge.source || edge.source == edge.target)
        return op->emitError("selected conversion does not match rebuilt source encoding");
      auto target = RankedTensorType::get(source.getShape(), source.getElementType(), edge.target);
      operands[edge.operand] = builder.create<ConvertLayoutOp>(op->getLoc(), target,
                                                              operands[edge.operand]);
    }
    SmallVector<Type> results;
    for (Value value : op->getResults()) {
      auto type = replacementType(value);
      if (failed(type)) return failure();
      results.push_back(*type);
    }
    SmallVector<Block *> successors;
    for (Block *successor : op->getSuccessors()) {
      auto *copy = mapping.lookupOrNull(successor);
      if (!copy) return op->emitError("unmapped successor during layout rebuilding");
      successors.push_back(copy);
    }
    Operation *copy = Operation::create(op->getLoc(), op->getName(), results,
        operands, op->getDiscardableAttrDictionary(), op->getPropertiesStorage(),
        successors, op->getNumRegions());
    block->push_back(copy);
    mapping.map(op, copy);
    mapping.map(op->getResults(), copy->getResults());
    if (Attribute layout = storage.lookup(op)) copy->setAttr("layout", layout);
    if (op->getName().getStringRef() == "arith.constant" && !results.empty() &&
        isa<RankedTensorType>(results.front())) {
      auto value = dyn_cast_or_null<DenseElementsAttr>(op->getAttr("value"));
      if (!value) return op->emitError("cannot retype non-dense tensor constant");
      copy->setAttr("value", value.reshape(cast<ShapedType>(results.front())));
    }
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      SmallVector<Type> inputs(function.getArgumentTypes());
      SmallVector<Type> outputs(function.getResultTypes());
      if (!function.isExternal()) {
        for (auto [index, arg] : llvm::enumerate(function.getArguments())) {
          auto type = replacementType(arg);
          if (failed(type)) return failure();
          inputs[index] = *type;
        }
      }
      for (auto [index, type] : llvm::enumerate(outputs)) {
        if (!isa<TensorType>(type)) continue;
        auto found = functionResults.find({op, unsigned(index)});
        if (found == functionResults.end())
          return op->emitError("missing function signature result layout");
        outputs[index] = found->second;
      }
      cast<func::FuncOp>(copy).setType(FunctionType::get(op->getContext(), inputs, outputs));
    }
    for (auto [oldRegion, newRegion] : llvm::zip(op->getRegions(), copy->getRegions()))
      if (failed(rebuildRegion(oldRegion, newRegion))) return failure();
    return success();
  }

  const LayoutConstraintGraph &graph;
  const LayoutSolution &solution;
  LayoutTypeConverter converter;
  IRMapping mapping;
  DenseMap<Value, Type> types;
  DenseMap<Operation *, Attribute> storage;
  DenseMap<std::pair<Operation *, unsigned>, Type> functionResults;
  DenseMap<Operation *, SmallVector<ConversionSnapshot>> conversions;
};
} // namespace

LogicalResult materializeDistributedLayouts(
    Operation *root, const LayoutConstraintGraph &graph,
    const LayoutSolution &solution, ArrayRef<LayoutConversionEdge> conversions) {
  auto module = dyn_cast<ModuleOp>(root);
  if (!module) return root->emitError("layout materialization requires a module root");
  if (conversions.size() != solution.conversions.size() ||
      !llvm::equal(conversions, solution.conversions, [](const auto &a, const auto &b) {
        return a.use == b.use && a.constraint == b.constraint &&
               a.sourceEncoding == b.sourceEncoding && a.targetEncoding == b.targetEncoding &&
               a.resolution == b.resolution;
      }))
    return root->emitError("conversion argument differs from selected layout solution");
  auto target = createSM90LayoutTarget();
  if (failed(verifySolvedLayoutGraph(graph, solution, *target, root->getLoc())))
    return failure();
  LayoutRebuild rebuild(graph, solution);
  if (failed(rebuild.prepare(root, conversions))) return failure();
  OwningOpRef<ModuleOp> staged(cast<ModuleOp>(root->cloneWithoutRegions()));
  if (failed(rebuild.rebuildRegion(module.getBodyRegion(), staged->getBodyRegion())) ||
      failed(verify(*staged)) || failed(verifyMaterializedLayouts(*staged, *target)))
    return failure();
  // This is the only original-IR mutation. No graph/solution identity may be
  // dereferenced after the old body (and original use pointers) is destroyed.
  module.getBodyRegion().takeBody(staged->getBodyRegion());
  return success();
}

LogicalResult materializeLayouts(Operation *root,
                                 const LayoutConstraintGraph &graph,
                                 const LayoutSolution &solution) {
  return materializeDistributedLayouts(root, graph, solution, solution.conversions);
}

LogicalResult verifyMaterializedLayouts(Operation *root,
                                        LayoutTarget &target) {
  if (failed(verify(root))) return failure();
  auto graph = collectLayoutConstraints(root, target, LayoutCollectionMode::RelationsOnly);
  if (failed(graph)) return failure();
  LayoutSolution actual;
  for (LayoutVar &var : graph->getVariables()) {
    Attribute encoding;
    if (var.kind == LayoutKind::Storage) {
      encoding = cast<LayoutViewOp>(var.anchor).getLayoutAttr();
      if (!encoding)
        return var.anchor->emitError("unresolved storage layout for materialized view");
    } else {
      Type type;
      if (var.value) type = var.value.getType();
      else if (var.use) type = var.use->get().getType();
      else if (var.functionResult)
        type = cast<func::FuncOp>(var.anchor).getResultTypes()[*var.functionResult];
      auto tensor = dyn_cast_or_null<RankedTensorType>(type);
      if (tensor) encoding = tensor.getEncoding();
      if (!isa_and_nonnull<DistributedEncodingAttr>(encoding))
        return emitError(var.anchor ? var.anchor->getLoc() : root->getLoc())
               << "unresolved distributed tensor encoding at materialized boundary";
    }
    // Pin synthetic uses to the encoding actually present on the operand.
    // Propagated alternatives must not hide a missing conversion by imagining
    // a future Convertible edge. No solver or propagation runs here.
    actual.assignments[var.id] = encoding;
    var.candidates = {{encoding, kInvalidProvenanceID, 0}};
    var.state = LayoutState::Resolved;
  }
  return verifySolvedLayoutGraph(*graph, actual, target, root->getLoc());
}

} // namespace mlir::frisk
