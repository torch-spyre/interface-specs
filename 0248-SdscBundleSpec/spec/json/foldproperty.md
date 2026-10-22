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
to describe how data is partitioned across cores, corelets, and time steps.
The `factor_` controls how many equal slices a dimension is divided into at
that level; the `label_` names the level so the fold engine knows which loop
index to use.

## Context

`FoldProperty` appears in one place:

| Parent field | Role |
|---|---|
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

A `FoldProperty` inside a `FoldManager.dim_prop_attr` for a core-level dimension split across 2 cores:

```json
{"factor_": 2, "label_": "core_fold"}
```

`factor_: 2` means the dimension is split into 2 slices — one per core. `factor_: 1` means no
splitting at that level (all cores see the same slice).

---

[↑ Spec Map](../../SuperDSC-Bundle.md#spec-map)|
