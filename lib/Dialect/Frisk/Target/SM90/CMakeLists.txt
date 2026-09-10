add_frisk_library(FriskSM90LayoutTarget
  SM90LayoutTarget.cpp
  SM90DistributedCandidates.cpp

  LINK_LIBS PUBLIC
  FriskAnalysis
  FriskIR
  MLIRIR
)
