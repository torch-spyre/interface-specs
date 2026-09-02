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
     ├─── dimToSymbolMappingOpcodeCorrection_: Object<symbol → string>
     ├─── inputSymbolsAndTags_: Object<symbol → string>
     ├─── symbolDefinitions_: Object
     ├─── debug_handle_: DebugHandle | null
     │    └─── DebugHandle
     │         ├─── id: string (required)
     │         ├─── source: SourceLoc | null (required)
     │         │    └─── SourceLoc
     │         │         ├─── file: string (required)
     │         │         ├─── start_line: integer (required)
     │         │         ├─── start_col: integer (required)
     │         │         ├─── end_line: integer | null (required)
     │         │         └─── end_col: integer | null (required)
     │         ├─── aten_op: string | null (required)
     │         ├─── ir_chain: Array<string> (required)
     │         ├─── fused_from: Array<DebugHandle> (required, recursive)
     │         └─── transform_history: Array<ProvenanceTransform> (required)
     │              └─── ProvenanceTransform
     │                   ├─── kind: "rewrite"|"fusion"|"decomposition"|"clone"|"remap" (required)
     │                   ├─── pass_name: string (required)
     │                   └─── reason: string | null (required)
     ├─── datadscs_: Array<Object>
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
                    │    └─── <dimension>_: number (e.g., mb_, out_, in_; -1 = unset)
                    │
                    ├─── dimToSymbolMapping_: Object<dimension → Array<string>>
                    ├─── coordinateMasking_: Object
                    ├─── maskingConstId_: integer (minimum: -1)
                    ├─── pdsRelation_: Object
                    │    └─── isPdsReuse: 0 | 1
                    │
                    ├─── dataStageParam_: Object<coreId → Object>
                    │    ├─── name_: "core" | "corelet" | "row"
                    │    ├─── ss_: DataStructDims
                    │    └─── el_: DataStructDims
                    │
                    ├─── primaryDsInfo_: Object<label → Object>
                    │    ├─── layoutDimOrder_: Array<string>
                    │    ├─── stickDimOrder_: Array<string>
                    │    ├─── stickSize_: Array<integer>
                    │    └─── stickRepl_: Array<integer>
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
                    │         ├─── isStartAddrSymbolic_: boolean | 0 | 1
                    │         ├─── startAddressCoreCorelet_: FoldManager
                    │         ├─── backGapCore_: Object
                    │         ├─── padding_: Object
                    │         ├─── indirectAllocType_: "no_indirection" | "value_tensor" | "index_tensor"
                    │         ├─── relatedIndirectAccessAlloc_: string
                    │         ├─── indexTensorType_: "index" | "address"
                    │         └─── coordinates_: Object
                    │              ├─── coordInfo: Object<dimension → CoordinateInfo>
                    │              │    └─── CoordinateInfo
                    │              │         ├─── spatial: integer (required)
                    │              │         ├─── temporal: integer (required)
                    │              │         ├─── elemArr: integer (required)
                    │              │         ├─── padding: "nopad" | "lowered_padded" | "padded_nozeropad"
                    │              │         │             | "padded_wzeropad" | "padded_fullspan"
                    │              │         │             | "padded_fullspan_wunneeded" (required)
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
                    │         ├─── wordLength: number (>0; fractional for sub-byte formats, e.g. 0.5)
                    │         ├─── dataFormat_: "SEN169_FP16" | "IEEE_FP32" | "IEEE_FP16" | "BFLOAT16"
                    │         │                | "BOOL" | "SEN143_FP8" | "SEN152_FP8" | "SEN153_FP9"
                    │         │                | "SEN18F_FP24" | "SEN080_FP8" | "SEN053_FP8"
                    │         │                | "SEN121_FP4" | "SENINT2" | "SENINT4" | "SENINT8"
                    │         │                | "SENINT16" | "SENINT24" | "SENUINT2" | "SENUINT32"
                    │         │                | "IEEE_INT32" | "IEEE_INT64" (required)
                    │         └─── memOrg_: MemoryOrganization (required)
                    │              ├─── hbm: MemorySlot
                    │              │    ├─── isPresent: 0 | 1 (required)
                    │              │    ├─── isPadded: 0 | 1
                    │              │    ├─── isZeroPadded: 0 | 1
                    │              │    ├─── dsOffset: integer
                    │              │    └─── allocateNode_: string
                    │              └─── lx: MemorySlot
                    │                   ├─── isPresent: 0 | 1 (required)
                    │                   ├─── isPadded: 0 | 1
                    │                   ├─── isZeroPadded: 0 | 1
                    │                   ├─── dsOffset: integer
                    │                   └─── allocateNode_: string
                    │
                    ├─── constantInfo_: Object<index → ConstantInfo> | string "{}"
                    │    └─── ConstantInfo (when object)
                    │         ├─── name_: string (required)
                    │         ├─── dataFormat_: string (required, same values as LabeledDataStructure.dataFormat_)
                    │         └─── data_: FoldManager (required)
                    │
                    └─── computeOp_: Array<ComputeOperation> (required)
                         │
                         └─── ComputeOperation
                              ├─── exUnit: "sfp" | "pt" (required)
                              ├─── opFuncName: string (required)
                              ├─── attributes_: Object
                              │    ├─── dataFormat_: string
                              │    └─── fidelity_: "regular" | "fast"
                              ├─── location: "Inner"
                              ├─── inputLabeledDs: Array<string> (required)
                              ├─── outputLabeledDs: Array<string> (required)
                              ├─── indirectAccessIndexLabeledDs: Array<string>
                              └─── interimLabeledDs: Array<string>
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
11. **debug_handle_**: Source-to-kernel provenance emitted by the frontend; `null` is a valid value (not missing data). Not read by the deeptools loader.
12. **DebugHandle**: Nestable provenance handle mirroring MLIR locations. `fused_from` is recursive.
13. **CoordinateInfo.padding**: Six specific values replacing the former `"pad"` catch-all — use `"padded_nozeropad"`, `"padded_wzeropad"`, `"padded_fullspan"`, `"padded_fullspan_wunneeded"`, or `"lowered_padded"` as appropriate.
14. **wordLength**: A `number` (not integer) to support sub-byte formats — 4-bit types carry `0.5`.
15. **indirectAllocType_**: Constrained to `"no_indirection"`, `"value_tensor"`, or `"index_tensor"`; when `"index_tensor"`, `indexTensorType_` must also be present.
16. **ComputeOperation.location**: Only `"Inner"` is emitted by the torch-spyre frontend (the loader accepts all 32 loop names).
17. **fidelity_**: Valid values are `"regular"` and `"fast"` (`"high"` and `"low"` are no longer valid).

---

| [← Previous: SDSC JSON API](SDSC-json-api.md) | [↑ Table of Contents](README.md) | [Next: SuperDsc Object →](superdsc-object.md) |
|:--|:--:|--:|
