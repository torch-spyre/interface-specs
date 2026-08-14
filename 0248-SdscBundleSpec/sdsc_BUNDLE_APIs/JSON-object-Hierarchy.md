# Object Hierarchy

The following diagram shows the complete object hierarchy of the sdscbundle-schema.json format:

```
Root Object (Dynamic operation name key, e.g., "exp", "matmul")
│
└─── SuperDsc (Bundle-level configuration)
     │
     ├─── sdscFoldProps_: Array<FoldProperty>
     │    └─── FoldProperty
     │         ├─── factor_: integer
     │         └─── label_: string
     │
     ├─── sdscFolds_: FoldManager
     │    ├─── dim_prop_func: Array<Object>
     │    │    └─── Const | Map | Affine | WkSplit
     │    ├─── dim_prop_attr: Array<FoldProperty>
     │    └─── data_: Object (coordinate-indexed values)
     │
     ├─── coreFoldProp_: FoldProperty (required)
     ├─── coreletFoldProp_: FoldProperty (required)
     ├─── numCoresUsed_: integer (required)
     ├─── coreIdToDsc_: Object<coreId → dscIndex> (required)
     ├─── numWkSlicesPerDim_: Object<dimension → sliceCount>
     ├─── coreIdToWkSlice_: Object<coreId → Object<dimension → sliceIndex>>
     ├─── coreIdToDscSchedule: Object<coreId → Array<[4 integers]>> (required)
     │
     └─── dscs_: Array<WrappedDesignSpaceConfig> (required)
          │
          └─── WrappedDesignSpaceConfig (operation name wrapper)
               │
               └─── DesignSpaceConfig (per-operation configuration)
                    │
                    ├─── numCoresUsed_: integer (required)
                    ├─── numCoreletsUsed_: integer
                    ├─── coreIdsUsed_: Array<integer> (required)
                    │
                    ├─── N_: DataStructDims (required)
                    │    ├─── name_: string
                    │    └─── <dimension>_: integer (e.g., mb_, out_, in_)
                    │
                    ├─── coordinateMasking_: Object
                    ├─── maskingConstId_: integer
                    │
                    ├─── dataStageParam_: Object<coreId → Object>
                    │    ├─── ss_: DataStructDims
                    │    └─── el_: DataStructDims
                    │
                    ├─── primaryDsInfo_: Object<label → Object>
                    │    ├─── layoutDimOrder_: Array<string>
                    │    ├─── stickDimOrder_: Array<string>
                    │    └─── stickSize_: Array<integer>
                    │
                    ├─── scheduleTree_: Array<ScheduleTreeNode> (required)
                    │    │
                    │    └─── ScheduleTreeNode
                    │         ├─── nodeType_: "allocate" (required)
                    │         ├─── name_: string (required)
                    │         ├─── prev_: string
                    │         ├─── ldsIdx_: integer
                    │         ├─── component_: "hbm" | "lx"
                    │         ├─── layoutDimOrder_: Array<string>
                    │         ├─── maxDimSizes_: Array<integer>
                    │         ├─── startAddressCoreCorelet_: FoldManager
                    │         ├─── backGapCore_: Object
                    │         └─── coordinates_: Object
                    │              ├─── coordInfo: Object<dimension → CoordinateInfo>
                    │              │    └─── CoordinateInfo
                    │              │         ├─── spatial: integer (required)
                    │              │         ├─── temporal: integer (required)
                    │              │         ├─── elemArr: integer (required)
                    │              │         ├─── padding: "nopad" | "pad" (required)
                    │              │         └─── folds: FoldManager (required)
                    │              └─── coreIdToWkSlice_: Object
                    │
                    ├─── labeledDs_: Array<LabeledDataStructure> (required)
                    │    │
                    │    └─── LabeledDataStructure
                    │         ├─── ldsIdx_: integer (required)
                    │         ├─── dsName_: string (required)
                    │         ├─── dsType_: "INPUT" | "OUTPUT" | "KERNEL" | "KERNEL_IDX" (required)
                    │         ├─── scale_: Array<number>
                    │         ├─── wordLength: integer
                    │         ├─── dataFormat_: "SEN169_FP16" | "SEN169_BFP16" | ... (required)
                    │         └─── memOrg_: Object (required)
                    │              ├─── hbm: {isPresent: 0|1}
                    │              └─── lx: {isPresent: 0|1}
                    │
                    ├─── constantInfo_: Object<index → ConstantInfo> | string "{}" (required)
                    │    └─── ConstantInfo (when object)
                    │         ├─── name_: string (required)
                    │         ├─── dataFormat_: string (required)
                    │         └─── data_: FoldManager (required)
                    │
                    └─── computeOp_: Array<ComputeOperation> (required)
                         │
                         └─── ComputeOperation
                              ├─── exUnit: "sfp" | "pt" (required)
                              ├─── opFuncName: string (required)
                              ├─── attributes_: Object
                              │    ├─── dataFormat_: string
                              │    └─── fidelity_: "regular" | "high" | "low"
                              ├─── location: "Inner" | "Outer"
                              ├─── inputLabeledDs: Array<string> (required)
                              └─── outputLabeledDs: Array<string> (required)
```

## Key Structural Points

1. **Root Level**: Dynamic operation name as key (e.g., "exp", "matmul", "softmax")
2. **SuperDsc**: Bundle-level configuration containing fold properties and array of DSCs
3. **WrappedDesignSpaceConfig**: Each DSC in the `dscs_` array is wrapped with operation name
4. **DesignSpaceConfig**: Core operation configuration with tensors, scheduling, and compute ops
5. **FoldManager**: Reusable structure for coordinate management at multiple levels
6. **dsType_ Values**: `INPUT`, `OUTPUT`, `KERNEL`, `KERNEL_IDX` (from torch_spyre constants)
7. **nodeType_ Values**: Currently only `allocate` (torch-spyre only generates allocation nodes)
8. **component_ Values**: `hbm`, `lx` (memory components used by torch-spyre)
9. **constantInfo_**: Object with numeric keys ("0", "1", etc.) mapping to ConstantInfo, or string `"{}"` when no constants
10. **ConstantInfo**: Generated by `generate_constant_info()` with name, dataFormat, and FoldManager-based data

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: Padding →](padding.md) |
|:--|:--:|--:|
