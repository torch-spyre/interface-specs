# DesignSpaceConfig

A `DesignSpaceConfig` (DSC) is the complete description of a single compute
configuration executed by one or more Spyre cores. It holds the total tensor
dimensions, the tensor descriptors, the memory allocation schedule, and the
compute operations to perform. The [`SuperDsc`](superdsc-object.md) object
wraps one or more DSCs in its `dscs_` array — typically one per operation,
with multiple entries only when work is unevenly divided across cores.

## Context

Each `DesignSpaceConfig` is the value inside a single-key
`{"<op_name>": <DesignSpaceConfig>}` entry in `SuperDsc.dscs_`. The
`coreIdToDsc_` map in `SuperDsc` assigns each core a zero-based index into
that array, determining which DSC each core executes.

```json
"dscs_": [
  {
    "gelu": { <DesignSpaceConfig> }
  }
]
```

## Structure

```json
{
  "numCoresUsed_":       <int>,
  "numCoreletsUsed_":    <int>,
  "coreIdsUsed_":        [<int>, ...],
  "N_":                  <DataStructDims>,
  "dimToSymbolMapping_": <map<string, array<string>>>,
  "coordinateMasking_":  <object>,
  "maskingConstId_":     <int>,
  "dataStageParam_":     <map<string, DataStageParam>>,
  "primaryDsInfo_":      <map<string, PrimaryDsInfo>>,
  "scheduleTree_":       [<ScheduleTreeNode>, ...],
  "labeledDs_":          [<LabeledDataStructure>, ...],
  "constantInfo_":       <map<string, ConstantInfo>> | "{}",
  "computeOp_":          [<ComputeOperation>, ...]
}
```

## Fields

Six fields are required. No additional properties are allowed.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `numCoresUsed_` | integer | Yes | >= 1 | Number of Spyre cores used by this DSC. Must match the length of `coreIdsUsed_`. |
| `numCoreletsUsed_` | integer | No | >= 1 | Number of corelets used. |
| `coreIdsUsed_` | array of integer | Yes | Each >= 0 | IDs of the cores that execute this DSC. Length must equal `numCoresUsed_`. |
| `N_` | [DataStructDims](datastructdims.md) | Yes | — | Total (un-tiled) tensor dimensions for this operation. |
| `dimToSymbolMapping_` | map&lt;string, array&lt;string&gt;&gt; | No | Keys: dim names | Mapping from dimension names to symbolic variable names for symbolic dimension support. |
| `coordinateMasking_` | object | No | — | Coordinate masking configuration for masked operations. |
| `maskingConstId_` | integer | No | — | Index into `constantInfo_` of the masking constant. |
| `dataStageParam_` | map&lt;string, object&gt; | No | Keys: `^[0-9]+$` (core ID) | Data staging parameters per core. Each value has `ss_` (steady-state dimensions) and `el_` (epilogue dimensions). See [DataStageParam](datastageparam.md). |
| `primaryDsInfo_` | map&lt;string, object&gt; | No | Keys: `^[A-Z_]+$` (dsType) | Per-tensor-type layout info. Each value has `layoutDimOrder_`, `stickDimOrder_`, and `stickSize_`. See [PrimaryDsInfo](primarydsinfo.md). |
| `scheduleTree_` | array of [ScheduleTreeNode](scheduletreenode.md) | Yes | — | Ordered list of memory allocation and compute schedule nodes. |
| `labeledDs_` | array of [LabeledDataStructure](labeleddatastructure.md) | Yes | — | Descriptors for all tensors (input and output) used by this DSC. |
| `constantInfo_` | map&lt;string, [ConstantInfo](constantinfo.md)&gt; or `"{}"` | No | Keys: `^[0-9]+$` | Named constants used by the operation. Set to the string `"{}"` when no constants are needed. |
| `computeOp_` | array of [ComputeOperation](computeoperation.md) | Yes | — | Compute operations to execute. More than one entry when operations are fused. |

## Example

A minimal 2-core GELU DSC showing required fields only:

```json
{
  "gelu": {
    "numCoresUsed_": 2,
    "numCoreletsUsed_": 2,
    "coreIdsUsed_": [0, 1],
    "N_": {"mb_": 32, "in_": 128},
    "labeledDs_": [
      {"ldsIdx_": 0, "dsName_": "gelu-Tensor0", "dsType_": "INPUT",  "dataFormat_": "SEN169_FP16", "memOrg_": {"hbm": {"isPresent": 1}, "lx": {"isPresent": 0}}},
      {"ldsIdx_": 1, "dsName_": "gelu-Tensor1", "dsType_": "OUTPUT", "dataFormat_": "SEN169_FP16", "memOrg_": {"hbm": {"isPresent": 1}, "lx": {"isPresent": 0}}}
    ],
    "scheduleTree_": [
      {"nodeType_": "allocate", "name_": "gelu-Tensor0", "ldsIdx_": 0, "component_": "hbm"},
      {"nodeType_": "allocate", "name_": "gelu-Tensor1", "ldsIdx_": 1, "component_": "hbm"}
    ],
    "constantInfo_": "{}",
    "computeOp_": [
      {
        "exUnit": "sfp",
        "opFuncName": "gelufwd",
        "attributes_": {"dataFormat_": "SEN169_FP16", "fidelity_": "regular"},
        "inputLabeledDs":  ["gelu-Tensor0-idx0"],
        "outputLabeledDs": ["gelu-Tensor1-idx1"]
      }
    ]
  }
}
```

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: DataStructDims →](datastructdims.md) |
|:--|:--:|--:|
