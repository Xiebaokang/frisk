add_frisk_library(FriskTransforms
    LayoutInfer.cpp
    MaterializeLayouts.cpp

    DEPENDS FriskTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    FriskLayoutAnalysis
    FriskIR
    FriskSM90LayoutTarget
)
