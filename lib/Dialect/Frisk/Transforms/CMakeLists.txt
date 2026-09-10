add_frisk_library(FriskTransforms
    LayoutInfer.cpp
    MaterializeLayouts.cpp
    LayoutTypeConverter.cpp
    OptimizeLayoutConversions.cpp

    DEPENDS FriskTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRTransforms
    FriskLayoutAnalysis
    FriskIR
    FriskSM90LayoutTarget
)
