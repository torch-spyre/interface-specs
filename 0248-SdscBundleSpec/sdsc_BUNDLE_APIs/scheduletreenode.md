# ScheduleTreeNode

A `ScheduleTreeNode` represents one step in the ordered execution plan for a
[`DesignSpaceConfig`](designspaceconfig.md). Each node either allocates a
tensor in a specific memory component or orchestrates a data-movement step.
Nodes are linked by their `prev_` field to form a dependency chain, and refer
to tensors by their zero-based index into `labeledDs_` via `ldsIdx_`.

## Context

`ScheduleTreeNode` entries appear in the `scheduleTree_` array inside each
[`DesignSpaceConfig`](designspaceconfig.md):

```json
"scheduleTree_": [
  { <ScheduleTreeNode> },
  { <ScheduleTreeNode> },
  ...
]
```

Tensor identity flows from `ldsIdx_` → [`LabeledDataStructure`](labeleddatastructure.md).
Start-address folding is expressed through a [`FoldManager`](foldmanager.md)
in `startAddressCoreCorelet_`. Spatial tiling coordinates are captured in a
[`CoordinateContainer`](coordinatecontainer.md) in `coordinates_`.

## Structure

```json
{
  "nodeType_":                 "allocate",
  "name_":                     "<string>",
  "prev_":                     "<string>",
  "ldsIdx_":                   <int>,
  "component_":                "hbm" | "lx",
  "layoutDimOrder_":           ["<dim>", ...],
  "maxDimSizes_":              [<int>, ...],
  "isStartAddrSymbolic_":      <bool | 0 | 1>,
  "startAddressCoreCorelet_":  <FoldManager>,
  "backGapCore_":              { "<dim>": { "<coreId>": "<gapStr>" } },
  "padding_":                  <object>,
  "indirectAllocType_":        "<string>",
  "relatedIndirectAccessAlloc_": "<string>",
  "coordinates_":              <CoordinateContainer>
}
```

## Fields

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `nodeType_` | string enum | Yes | `"allocate"` | Type of schedule node. Currently only `"allocate"` is supported. |
| `name_` | string | Yes | — | Unique identifier for this node within the `scheduleTree_` array. Referenced by other nodes via `prev_`. |
| `prev_` | string | No | Must match another node's `name_` | Name of the predecessor node, establishing execution ordering. Omit for the first node. |
| `ldsIdx_` | integer | No | >= 0 | Zero-based index into the parent `labeledDs_` array identifying which tensor this node allocates. See [`LabeledDataStructure`](labeleddatastructure.md). |
| `component_` | string enum | No | `"hbm"` or `"lx"` | Target memory component for the allocation. `"hbm"` = high-bandwidth memory; `"lx"` = LX scratchpad. |
| `layoutDimOrder_` | array of string | No | Dimension names declared in `DataStructDims` | Physical memory layout order for this allocation, outermost first. Overrides the global `primaryDsInfo_` order when present. |
| `maxDimSizes_` | array of integer | No | Parallel to `layoutDimOrder_`; each integer >= 1 | Maximum element count per dimension. Used to size the allocated buffer. |
| `isStartAddrSymbolic_` | boolean or integer (0/1) | No | `true`/`false` or `1`/`0` | When true, start addresses are expressed as symbolic values rather than concrete byte offsets. Written by the backend compiler when symbolic addressing is required. |
| `startAddressCoreCorelet_` | [FoldManager](foldmanager.md) | No | — | Per-core / per-corelet start address for this allocation, expressed as a FoldManager. See [FoldManager](foldmanager.md) for the full encoding. |
| `backGapCore_` | `map<dim, map<coreId, string>>` | No | Written conditionally when `tensor.backGap` is true | Back-gap size (in elements) per dimension per core. Outer key is a dimension name (`^[A-Za-z_][A-Za-z0-9_]*$`); inner key is a core ID (`^-?[0-9]+$`); value is the gap size as a decimal string. |
| `padding_` | object | No | — | Padding parameters for this allocation. Written by the backend compiler when the tensor requires memory padding beyond its logical size. |
| `indirectAllocType_` | string | No | — | Indirect allocation type tag. Present when this node participates in an indirect-access allocation pattern. |
| `relatedIndirectAccessAlloc_` | string | No | Must match another node's `name_` | Name of the related indirect-access allocation node, linking this node to its paired indirect access entry. |
| `coordinates_` | [CoordinateContainer](coordinatecontainer.md) | No | — | Spatial tiling coordinates for this allocation. Groups per-dimension [`CoordinateInfo`](coordinateinfo.md) alongside the `coreIdToWkSlice_` mapping. |

## Example

The example below shows two linked allocation nodes: a weight tensor placed in
HBM whose start address is pre-computed by core/corelet, and an activation
tensor in LX scratchpad with tiling coordinates.

```json
"scheduleTree_": [
  {
    "nodeType_":    "allocate",
    "name_":        "weight_alloc",
    "ldsIdx_":      1,
    "component_":   "hbm",
    "layoutDimOrder_": ["out", "in"],
    "maxDimSizes_":    [128, 64],
    "isStartAddrSymbolic_": false,
    "startAddressCoreCorelet_": {
      "dim_prop_func": [{"Map": {}}, {"Const": {}}],
      "dim_prop_attr": [
        {"factor_": 2, "label_": "core"},
        {"factor_": 2, "label_": "corelet"}
      ],
      "data_": {
        "[0, 0]": "0",
        "[1, 0]": "262144"
      }
    }
  },
  {
    "nodeType_":  "allocate",
    "name_":      "input_alloc",
    "prev_":      "weight_alloc",
    "ldsIdx_":    0,
    "component_": "lx",
    "layoutDimOrder_": ["mb", "out", "in"],
    "maxDimSizes_":    [32, 128, 64],
    "coordinates_": {
      "coordInfo": {
        "mb": {
          "spatial":  0,
          "temporal": 0,
          "elemArr":  0,
          "padding":  "nopad",
          "folds": {
            "dim_prop_func": [{"Const": {}}],
            "dim_prop_attr": [{"factor_": 32, "label_": "mb"}],
            "data_": { "[0]": 0 }
          }
        }
      },
      "coreIdToWkSlice_": {
        "0": {"mb": 0},
        "1": {"mb": 1},
        "2": {"mb": 2}
      }
    }
  }
]
```

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: CoordinateContainer →](coordinatecontainer.md) |
|:--|:--:|--:|
