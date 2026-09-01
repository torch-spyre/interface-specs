# FoldManager

A `FoldManager` maps a tuple of loop indices (core, corelet, time step, …) to a
concrete value such as a memory address, coordinate offset, or constant. It is
the composite expression built from [`FoldProperty`](foldproperty.md) entries —
see that page for the conceptual overview of folding.

Each dimension in the fold has a **function** (`dim_prop_func`) that says *how*
the index maps to a value, and an **attribute** (`dim_prop_attr`, a
[`FoldProperty`](foldproperty.md)) that says *which* fold level the dimension
belongs to and how many slices it has. The optional `data_` map holds the actual
values, keyed by fold-coordinate tuples like `"[0, 1]"`.

## Where Folding Appears

`FoldManager` is used in five places across the schema:

| Parent field | Purpose |
|---|---|
| `SuperDsc.sdscFolds_` | Bundle-level fold addresses / mappings |
| `ScheduleTreeNode.startAddressCoreCorelet_` | Per-core tensor start addresses |
| `CoordinateInfo.folds` | Per-dimension coordinate computation |
| `ConstantInfo.data_` | Per-core constant values |
| `FoldManager.data_` *(self)* | Nested fold data values |

For details on how `startAddressCoreCorelet_` and `coordinates_` are filled in
practice, see [ScheduleTreeNode](scheduletreenode.md).

## Structure

```json
{
  "dim_prop_func": [
    { "Const": {} }
    | { "Map": {} }
    | { "Affine": {"alpha_": <int>, "beta_": <int>} }
    | { "WkSplit": {} },
    ...
  ],
  "dim_prop_attr": [
    <FoldProperty>,
    ...
  ],
  "data_": {
    "[<i>, <j>, ...]": <string> | <integer> | [<integer>, ...]
  }
}
```

## Fields

`dim_prop_func` and `dim_prop_attr` are required and must have the same
length — one entry per fold dimension. No additional properties are allowed.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `dim_prop_func` | array of object | Yes | Same length as `dim_prop_attr` | One fold function per dimension. Each element is a single-key object whose key is the function type. See [Function Types](#function-types). |
| `dim_prop_attr` | array of [FoldProperty](foldproperty.md) | Yes | Same length as `dim_prop_func` | One [`FoldProperty`](foldproperty.md) per dimension, giving the fold level label and factor. |
| `data_` | object | No | Keys: `^\\[[0-9, ]+\\]$` | Lookup table keyed by fold-coordinate tuples (e.g. `"[0, 1]"`). Values are strings, integers, or arrays of integers depending on context. |

## Function Types

Each element of `dim_prop_func` is a single-key object. The key determines
how the fold index for that dimension is resolved:

| Type | Structure | Behaviour |
|---|---|---|
| `Const` | `{"Const": {}}` | The value is the same for all indices at this dimension level. The single entry in `data_` (keyed `"[0]"` or `"[0, 0]"`) is used regardless of the loop index. |
| `Map` | `{"Map": {}}` | The value is looked up in `data_` using the fold-coordinate tuple. Each combination of indices maps to a distinct entry. Used for per-core start addresses. |
| `Affine` | `{"Affine": {"alpha_": <int>, "beta_": <int>}}` | The value is computed as `alpha_ * index + beta_`. No `data_` lookup needed. Used for coordinates that follow a linear stride. |
| `WkSplit` | `{"WkSplit": {}}` | The value is the work-slice index assigned to this core for this dimension. Resolved at runtime from `coreIdToWkSlice_`. |

## Example

### startAddressCoreCorelet_ — per-core memory addresses via Map

Two cores with different start addresses; the corelet level is constant.
`[core, corelet]` tuples key into `data_`.

```json
"startAddressCoreCorelet_": {
  "dim_prop_func": [
    {"Map":   {}},
    {"Const": {}}
  ],
  "dim_prop_attr": [
    {"factor_": 2, "label_": "core"},
    {"factor_": 1, "label_": "corelet"}
  ],
  "data_": {
    "[0, 0]": "0",
    "[1, 0]": "131072"
  }
}
```

Core 0 starts at address `0`; core 1 starts at `131072` (128 KiB offset).
The corelet dimension uses `Const` so all corelets on a given core share
the same base address.

### CoordinateInfo.folds — affine coordinate stride

A dimension split 4 ways across cores with a stride of 32 elements, no
corelet or time-step splitting.

```json
"folds": {
  "dim_prop_func": [
    {"Affine": {"alpha_": 32, "beta_": 0}},
    {"Affine": {"alpha_": 0,  "beta_": 0}},
    {"Affine": {"alpha_": 1,  "beta_": 0}}
  ],
  "dim_prop_attr": [
    {"factor_": 4, "label_": "core_fold"},
    {"factor_": 1, "label_": "corelet_fold"},
    {"factor_": 32, "label_": "elem_arr_0"}
  ]
}
```

Core index `i` maps to coordinate `32 * i`. The corelet dimension has
`alpha_: 0` (no contribution). The element-array level has `alpha_: 1`
(element index passes through unchanged).

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: Stick Layout Constraints →](stick-layout-constraints.md) |
|:--|:--:|--:|
