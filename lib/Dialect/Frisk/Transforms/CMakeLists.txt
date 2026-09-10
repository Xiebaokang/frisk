add_frisk_library(FriskTransforms
    LayoutInfer.cpp
    MaterializeLayouts.cpp
    LayoutTypeConverter.cpp

    DEPENDS FriskTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    FriskLayoutAnalysis
    FriskIR
    FriskSM90LayoutTarget
)
