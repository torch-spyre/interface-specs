# FoldProperty

A `FoldProperty` pairs a **fold factor** with a **label** that identifies a
level of the Spyre memory hierarchy. It is the building block used by
[`FoldManager`](foldmanager.md) (in `dim_prop_attr`) and by
[`SuperDsc`](superdsc-object.md) (in `coreFoldProp_`, `coreletFoldProp_`, and
`sdscFoldProps_`) to describe how data is partitioned across cores, corelets,
and time steps. The `factor_` controls how many equal slices a dimension is
divided into at that level; the `label_` names the level so the fold engine
knows which loop index to use.

## Context

`FoldProperty` appears in three places:

| Parent field | Role |
|---|---|
| `SuperDsc.coreFoldProp_` | Defines the fold factor at the **core** level for the whole bundle |
| `SuperDsc.coreletFoldProp_` | Defines the fold factor at the **corelet** level for the whole bundle |
| `SuperDsc.sdscFoldProps_` | Optional array for additional bundle-level fold dimensions |
| `FoldManager.dim_prop_attr` | Per-dimension fold attributes inside a [`FoldManager`](foldmanager.md) |

For the conceptual explanation of how folding works, see [Folding](folding.md).

## Structure

```json
{
  "factor_": <int>,
  "label_":  "<string>"
}
```

## Fields

Both fields are required. No additional properties are allowed.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `factor_` | integer | Yes | >= 1 | Number of equal slices the dimension is divided into at this fold level. A factor of `1` means no splitting at that level. |
| `label_` | string | Yes | — | Name of the memory hierarchy level. Common values: `"core"`, `"corelet"`, `"time"`, `"core_fold"`, `"corelet_fold"`, `"row_fold"`, `"elem_arr_0"`, `"elem_arr_1"`. |

## Example

The two required `SuperDsc` fold properties for a 2-core, 2-corelet bundle:

```json
"coreFoldProp_":    {"factor_": 2, "label_": "core"},
"coreletFoldProp_": {"factor_": 2, "label_": "corelet"}
```

`factor_: 2` at the core level means each tensor dimension assigned to this
fold is split into 2 slices — one per core. `factor_: 1` would mean no
splitting at that level (all cores see the same slice).

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: FoldManager →](foldmanager.md) |
|:--|:--:|--:|
