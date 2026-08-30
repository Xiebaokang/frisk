#include "Dialect/Frisk/Analysis/LayoutCommon.h"
#include "Dialect/Frisk/IR/FriskLayoutInterfaces.h"

#include <type_traits>
#include <utility>

#include "gtest/gtest.h"

namespace mlir::frisk {
namespace {

static_assert(std::is_enum_v<LayoutKind>);
static_assert(std::is_enum_v<ProofStatus>);
static_assert(std::is_class_v<LayoutMapAttrInterface>);
static_assert(std::is_class_v<LayoutConstraintOpInterface>);
static_assert(std::is_same_v<
              decltype(std::declval<LayoutEncodingAttrInterface>().getKind()),
              LayoutKind>);

TEST(LayoutInterfaceTest, PublicKindsAreEnums) {
  EXPECT_NE(LayoutKind::Distributed, LayoutKind::Storage);
  EXPECT_NE(ProofStatus::Proven, ProofStatus::Unknown);
}

} // namespace
} // namespace mlir::frisk
