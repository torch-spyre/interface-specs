# ScheduleTreeNode

A `ScheduleTreeNode` represents one step in the ordered execution plan for a
[`DesignSpaceConfig`](designspaceconfig.md). Each node either allocates a
tensor in a specific memory component or orchestrates a data-movement step.
Nodes are linked by their `prev_` field to form a dependency chain, and refer
to tensors by their zero-based index into `labeledDs_` via `ldsIdx_`.

The schedule tree is a list of nodes that can be of types BLOCK, LOOP,
TRANSFER, COMPUTE, SYNC, CONDITION, ALLOCATE, and STICKMASK, among others.
Only `ALLOCATE` nodes need to be filled in from the front end — one per tensor
in `labeledDs_`.

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
  "indexTensorType_":          "index" | "address",
  "coordinates_":              <CoordinateContainer>
}
```

## Fields

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `nodeType_` | string enum | Yes | `"allocate"` | Type of schedule node. Always set to `"allocate"` for front-end-filled nodes. |
| `name_` | string | Yes | — | Unique identifier for this node within the `scheduleTree_` array. Referenced by other nodes via `prev_`. |
| `prev_` | string | No | Must match another node's `name_` | Name of the predecessor node, establishing execution ordering. Omit for the first node. |
| `ldsIdx_` | integer | No | >= 0 | Zero-based index into the parent `labeledDs_` array identifying which tensor this node allocates. See [`LabeledDataStructure`](labeleddatastructure.md). |
| `component_` | string enum | No | `"hbm"` or `"lx"` | Target memory component. `"hbm"` = high-bandwidth memory; `"lx"` = LX scratchpad. |
| `layoutDimOrder_` | array of string | No | Dimension names declared in `DataStructDims` | Physical memory layout order, outermost first. Tensor layout is specified from inner to outer. Overrides the global `primaryDsInfo_` order when present. |
| `maxDimSizes_` | array of integer | No | Parallel to `layoutDimOrder_` | Maximum element count per dimension. No minimum is enforced; `-1` is the typical value when the size is not constrained. |
| `isStartAddrSymbolic_` | boolean or integer (0/1) | No | `true`/`false` or `1`/`0` | When true (`1`), the `data_` values in `startAddressCoreCorelet_` are symbolic identifiers rather than concrete byte offsets. See [`startAddressCoreCorelet_`](#startaddresscorecorelet_) below. |
| `startAddressCoreCorelet_` | [FoldManager](foldmanager.md) | No | — | Per-core / per-corelet start address for this allocation, expressed as a FoldManager. See [`startAddressCoreCorelet_`](#startaddresscorecorelet_) below. |
| `backGapCore_` | `map<dim, map<coreId, string>>` | No | Written conditionally when `tensor.backGap` is true | Back-gap size (in elements) per dimension per core. Outer key is a dimension name (`^[A-Za-z_][A-Za-z0-9_]*$`); inner key is a core ID (`^-?[0-9]+$`, where `"-1"` denotes HBM); value is the gap size as a decimal string. |
| `padding_` | object | No | — | Padding type for each padded dimension. One of `NOPAD`, `LOWERED_PADDED`, `PADDED_NOZEROPAD`, `PADDED_WZEROPAD`, `PADDED_FULLSPAN`, `PADDED_FULLSPAN_WUNNEEDED`. See [Padding](padding.md). |
| `indirectAllocType_` | string | No | — | One of `no_indirection`, `index_tensor`, or `value_tensor`. `no_indirection` = ordinary direct allocation; `index_tensor` = holds indices for a gather/scatter; `value_tensor` = holds the data gathered/scattered via those indices. |
| `relatedIndirectAccessAlloc_` | string | No | Must match another node's `name_`; non-empty only when `indirectAllocType_` is `"index_tensor"` | Name of the related allocation node that provides actual data for the indirect reference. |
| `indexTensorType_` | string enum | No | `"index"` or `"address"`; only present when `indirectAllocType_` is `"index_tensor"` | How the index tensor's elements are interpreted: `"index"` = element indices; `"address"` = precomputed byte addresses. |
| `coordinates_` | [CoordinateContainer](coordinatecontainer.md) | No | — | Spatial tiling coordinates for this allocation. Groups per-dimension [`CoordinateInfo`](coordinateinfo.md) alongside the `coreIdToWkSlice_` mapping. See [`coordinates_`](#coordinates_) below. |

## startAddressCoreCorelet_

`startAddressCoreCorelet_` stores the starting memory address for each core and
corelet for this tensor allocation, expressed as a [`FoldManager`](foldmanager.md).

The typical pattern uses `Map` for the core dimension (each core may have a
different start address) and `Const` for the corelet and time dimensions (all
corelets on a core share the same base address):

```json
"startAddressCoreCorelet_": {
  "dim_prop_func": [
    {"Map":   {}},
    {"Const": {}},
    {"Const": {}}
  ],
  "dim_prop_attr": [
    {"factor_": 20, "label_": "core"},
    {"factor_": 1,  "label_": "corelet"},
    {"factor_": 1,  "label_": "time"}
  ],
  "data_": {
    "[0, 0, 0]":  "0",   "[1, 0, 0]":  "0",   "[2, 0, 0]":  "0",
    "[3, 0, 0]":  "0",   "[4, 0, 0]":  "0",   "[5, 0, 0]":  "128",
    "[6, 0, 0]":  "128", "[7, 0, 0]":  "128", "[8, 0, 0]":  "128",
    "[9, 0, 0]":  "128", "[10, 0, 0]": "256", "[11, 0, 0]": "256",
    "[12, 0, 0]": "256", "[13, 0, 0]": "256", "[14, 0, 0]": "256",
    "[15, 0, 0]": "384", "[16, 0, 0]": "384", "[17, 0, 0]": "384",
    "[18, 0, 0]": "384", "[19, 0, 0]": "384"
  }
}
```

This example corresponds to a tensor with a 4-way split. Each of the 4 slices
is consumed by 5 cores, so the start address is the same for 5 consecutive
cores. Each slice is 128 bytes, so successive slice start addresses are 128
bytes apart.

Start addresses in `data_` can be **symbolic**: instead of a concrete byte
offset, a symbolic identifier is used as the value and is substituted just
before the job is launched. Symbolic bindings are declared in the bundle MLIR
file — see [MLIR Bundle API](MLIR-bundle-API.md). Set `isStartAddrSymbolic_`
to `true` / `1` when symbolic addresses are used.

## coordinates_

`coordinates_` is a [`CoordinateContainer`](coordinatecontainer.md) whose
`coordInfo` map holds one [`CoordinateInfo`](coordinateinfo.md) entry per
tensor dimension. Each entry describes how that dimension is progressively split
across cores, corelets, rows, and the final leaf entities via its `folds`
[`FoldManager`](foldmanager.md).

### CoordinateInfo fields

| Field | Values | Description |
|---|---|---|
| `spatial` | typically `3` | Number of spatial split levels (core, corelet, row). |
| `temporal` | `0` | Set to `0` by the frontend. |
| `elemArr` | `1` or `2` | `1` for non-stick dimensions; `2` for stick dimensions. |
| `padding` | `"nopad"`, `"lowered_padded"`, `"padded_nozeropad"`, `"padded_wzeropad"`, `"padded_fullspan"`, or `"padded_fullspan_wunneeded"` | Padding state for this dimension. `"nopad"` is the common case; see [CoordinateInfo](coordinateinfo.md) for the full enum and [Padding](padding.md) for semantics. |
| `folds` | [FoldManager](foldmanager.md) | Encodes the dimension split hierarchy. |

### folds hierarchy

The `dim_prop_attr` labels in `coordinates_.coordInfo.<dim>.folds` follow a
fixed split hierarchy:

| Label | Level |
|---|---|
| `core_fold` | Split across cores |
| `corelet_fold` | Split across corelets within a core |
| `row_fold` | Split across rows within a corelet |
| `elem_arr_0` | Element count in the innermost (leaf) slice |
| `elem_arr_1` | Number of sticks per slice (stick dimensions only) |

The general structure is:

```json
"coordinates_": {
  "coordInfo": {
    "<dim_name>": {
      "spatial":  3,
      "temporal": 0,
      "elemArr":  1,
      "padding":  "nopad",
      "folds": {
        "dim_prop_func": [
          {"Affine": {"alpha_": <cores spanned per core-wise split>, "beta_": 0}},
          {"Affine": {"alpha_": 1, "beta_": 0}},
          {"Affine": {"alpha_": 1, "beta_": 0}},
          {"Affine": {"alpha_": <size of leaf slice>, "beta_": 0}}
        ],
        "dim_prop_attr": [
          {"factor_": <number of core-wise splits>,    "label_": "core_fold"},
          {"factor_": <number of corelet splits>,      "label_": "corelet_fold"},
          {"factor_": <number of row splits>,          "label_": "row_fold"},
          {"factor_": <number of elements per slice>,  "label_": "elem_arr_0"}
        ]
      }
    }
  }
}
```

For a **stick dimension** `coordInfo.elemArr` is `2` and `dim_prop_attr`
includes an additional `elem_arr_1` entry. The `factor_` of `elem_arr_0` gives
the number of elements per stick; the `factor_` of `elem_arr_1` gives the
number of sticks in the slice.

### Worked example

Consider a tensor with dimensions `x`, `in`, and `out` of sizes `[4, 2880, 2880]`.
The `x` dimension has a 4-way split across cores; `out` has a 5-way split
across cores; `in` is not split. Total: 20 splits, one per core.

#### Dimension **x** (4-way core split, non-stick)

```json
"folds": {
  "dim_prop_func": [
    {"Affine": {"alpha_": 1, "beta_": 0}},
    {"Affine": {"alpha_": 0, "beta_": 0}},
    {"Affine": {"alpha_": 0, "beta_": 0}},
    {"Affine": {"alpha_": 1, "beta_": 0}}
  ],
  "dim_prop_attr": [
    {"factor_": 4, "label_": "core_fold"},
    {"factor_": 1, "label_": "corelet_fold"},
    {"factor_": 1, "label_": "row_fold"},
    {"factor_": 1, "label_": "elem_arr_0"}
  ]
}
```

#### Dimension **out** (5-way core split, stick dimension)

`coordInfo.elemArr` is `2`. `dim_prop_attr` includes `elem_arr_1` for the
number of sticks per slice.

```json
"folds": {
  "dim_prop_func": [
    {"Affine": {"alpha_": 1, "beta_": 0}},
    {"Affine": {"alpha_": 0, "beta_": 0}},
    {"Affine": {"alpha_": 0, "beta_": 0}},
    {"Affine": {"alpha_": 1, "beta_": 0}},
    {"Affine": {"alpha_": 1, "beta_": 0}}
  ],
  "dim_prop_attr": [
    {"factor_": 5,   "label_": "core_fold"},
    {"factor_": 1,   "label_": "corelet_fold"},
    {"factor_": 1,   "label_": "row_fold"},
    {"factor_": 64,  "label_": "elem_arr_0"},
    {"factor_": 9,   "label_": "elem_arr_1"}
  ]
}
```

`elem_arr_0 factor_ = 64` → 64 elements per stick.
`elem_arr_1 factor_ = 9` → 9 sticks per core slice (9 × 64 = 576 = 2880 / 5).

#### Dimension **in** (not split)

```json
"folds": {
  "dim_prop_func": [
    {"Affine": {"alpha_": 2880, "beta_": 0}},
    {"Affine": {"alpha_": 0,    "beta_": 0}},
    {"Affine": {"alpha_": 0,    "beta_": 0}},
    {"Affine": {"alpha_": 1,    "beta_": 0}}
  ],
  "dim_prop_attr": [
    {"factor_": 1,    "label_": "core_fold"},
    {"factor_": 1,    "label_": "corelet_fold"},
    {"factor_": 1,    "label_": "row_fold"},
    {"factor_": 2880, "label_": "elem_arr_0"}
  ]
}
```

`core_fold factor_ = 1` confirms no core-level splitting. `alpha_ = 2880` at
the core level means the full 2880-element span is the coordinate stride passed
to the fold engine.

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

| [← Previous: DataStageParam](datastageparam.md) | [↑ Table of Contents](README.md) | [Next: CoordinateContainer →](coordinatecontainer.md) |
|:--|:--:|--:|
