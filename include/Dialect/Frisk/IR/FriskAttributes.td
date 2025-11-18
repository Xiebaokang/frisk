#ifndef FRISK_ATTRIBUTES_TD
#define FRISK_ATTRIBUTES_TD

include "Dialect/Frisk/IR/FriskDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "Dialect/Frisk/IR/FriskEnums.td"

def MemorySpaceAttr : EnumAttr<Frisk_Dialect, MemorySpaceEnum, "memory_space"> {
    let cppNamespace = "::mlir::frisk";
}

def GemmWarpPolicyAttr : EnumAttr<Frisk_Dialect, GemmWarpPolicyEnum, "gemm_warp_policy"> {
    let cppNamespace = "::mlir::frisk";
    let summary = "GEMM warp partitioning policy";
}

class Frisk_LayoutAttr : AttrDef<Frisk_Dialect, "layout"> {
    let cppNamespace = "::mlir::frisk";

    let summary = "";

    let storageNamespace = "detail";
    let storageType = "LayoutAttrStorage";
    let returnType = "::mlir::frisk::LayoutAttr";
    let convertFromStorage = "$_storage";

    let parameters = (ins
        AttrParameter<"ArrayAttr", "input_shape">,
        AttrParameter<"Attribute", "forward_index">,

        AttrParameter<"Attribute", "forward_thread", /*defaultValue=*/"nullptr">,
        AttrParameter<"Attribute", "replicate_size", /*defaultValue=*/"nullptr">     
    );

    let extraClassDefinition = [{
        bool isFragment() const {
            return getForwardThread() != nullptr;
        } 
    }];
}



#endif // FRISK_ATTRIBUTES_TD