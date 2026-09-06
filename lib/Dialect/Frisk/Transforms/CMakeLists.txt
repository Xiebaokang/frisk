add_frisk_library(FriskTransforms
    LayoutInfer.cpp

    DEPENDS FriskTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    FriskLayoutAnalysis
    FriskIR
    FriskSM90LayoutTarget
)
