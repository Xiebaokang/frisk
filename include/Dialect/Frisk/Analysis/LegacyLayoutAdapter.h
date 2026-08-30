#ifndef FRISK_ANALYSIS_LEGACYLAYOUTADAPTER_H
#define FRISK_ANALYSIS_LEGACYLAYOUTADAPTER_H

#include "Dialect/Frisk/IR/FriskAttributes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::frisk {

FailureOr<DistributedEncodingAttr>
convertLegacyDistributed(LayoutAttr legacy, ShapedType type, Location loc);

FailureOr<StorageLayoutAttr>
convertLegacyStorage(LayoutAttr legacy, MemRefType type, Location loc);

/// Exhaustive, bounded semantic checks used by the migration gate. They emit
/// the first mismatching logical or carrier coordinate at `loc`.
LogicalResult verifyLegacyDistributedEquivalent(
    LayoutAttr legacy, DistributedEncodingAttr converted, ShapedType type,
    Location loc);

LogicalResult verifyLegacyStorageEquivalent(LayoutAttr legacy,
                                            StorageLayoutAttr converted,
                                            MemRefType type, Location loc);

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LEGACYLAYOUTADAPTER_H
