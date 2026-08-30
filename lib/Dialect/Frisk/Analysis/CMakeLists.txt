add_frisk_library(FriskAnalysis
  LayoutAlgebra.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

# The migration adapter depends on both the standalone algebra and concrete
# Frisk attrs, so keep it out of FriskAnalysis to avoid an FriskIR cycle.
add_frisk_library(FriskLegacyLayoutAdapter
  LegacyLayoutAdapter.cpp

  LINK_LIBS PUBLIC
  FriskAnalysis
  FriskIR
  MLIRIR
)
