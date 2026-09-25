# FoldProperty

Folding is the mechanism by which a single parameterised SDSC artifact describes
the behaviour of all active cores without duplicating data for every core, corelet,
or time step. Instead of storing separate values for each combination of indices,
the fold system encodes a compact expression — constant, lookup table, or affine
formula — that is evaluated at compile time or resolved just-in-time before the
job is launched.

`FoldProperty` is the primitive building block of the fold system. It pairs a
**fold factor** with a **label** that identifies a level of the Spyre memory
hierarchy. It is used by [`FoldManager`](foldmanager.md) (in `dim_prop_attr`)
and by [`SuperDsc`](superdsc-object.md) (in `coreFoldProp_`, `coreletFoldProp_`,
and `sdscFoldProps_`) to describe how data is partitioned across cores, corelets,
and time steps. The `factor_` controls how many equal slices a dimension is
divided into at that level; the `label_` names the level so the fold engine
knows which loop index to use.

## Context

`FoldProperty` appears in four places:

| Parent field | Role |
|---|---|
| `SuperDsc.coreFoldProp_` | Defines the fold factor at the **core** level for the whole bundle |
| `SuperDsc.coreletFoldProp_` | Defines the fold factor at the **corelet** level for the whole bundle |
| `SuperDsc.sdscFoldProps_` | Optional array for additional bundle-level fold dimensions |
| `FoldManager.dim_prop_attr` | Per-dimension fold attributes inside a [`FoldManager`](foldmanager.md) |

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

## Folds Hierarchy

The `dim_prop_attr` labels in coordinate folds (such as `coordinates_.coordInfo.<dim>.folds`) follow a standard memory-hierarchy split:

| Label | Level / Role | Description |
|---|---|---|
| `core_fold` | Core level | Split across Spyre cores |
| `corelet_fold` | Corelet level | Split across corelets within a core |
| `row_fold` | Row level | Split across rows within a corelet |
| `elem_arr_0` | Leaf element slice | Number of contiguous elements per innermost slice (or per stick) |
| `elem_arr_1` | Stick slice | Number of sticks per slice (used for stick dimensions where `elemArr` is 2) |

**Note:** The standard hierarchy labels above (`core_fold`, `corelet_fold`, `row_fold`, `elem_arr_0`, `elem_arr_1`) apply when a dimension is partitioned across the spatial hierarchy. When a dimension is **collapsed** (not split across the hierarchy), a bare dimension-name label such as `"mb"` or `"kb"` may appear instead. In that case `spatial` and `elemArr` in the parent `CoordinateInfo` are both `0`, and the `FoldManager` typically uses a single `Const` function.

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

| [← Previous: SuperDsc Object](superdsc-object.md) | [↑ Table of Contents](README.md) | [Next: FoldManager →](foldmanager.md) |
|:--|:--:|--:|
