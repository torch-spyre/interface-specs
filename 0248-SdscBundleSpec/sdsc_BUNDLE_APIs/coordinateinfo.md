# CoordinateInfo

Coordinate information for a single dimension within a tensor allocation. `CoordinateInfo` objects
appear as values inside [`CoordinateContainer.coordInfo`](coordinatecontainer.md), keyed by
dimension name (e.g. `"mb"`, `"kb"`).

Each `CoordinateInfo` encodes how a dimension is positioned in the Spyre memory hierarchy
(spatial / temporal / element-array levels), whether it is padded, and how its address or
index is computed via a [`FoldManager`](foldmanager.md).

## Required Fields

All five fields are required. No additional properties are allowed.

## Structure

```json
{
  "spatial":  <int>,
  "temporal": <int>,
  "elemArr":  <int>,
  "padding":  "nopad" | "lowered_padded" | "padded_nozeropad" | "padded_wzeropad" | "padded_fullspan" | "padded_fullspan_wunneeded",
  "folds":    <FoldManager>
}
```

## Fields

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `spatial` | integer | Yes | >= 0 | Number of spatial split levels (e.g. core, corelet, row). Set to `3` when the dimension is split across cores, corelets, and rows; or `0` when collapsed or unpartitioned across the spatial hierarchy. |
| `temporal` | integer | Yes | >= 0 | Number of temporal split levels for LX scratchpad staging passes. Set to `0` by the frontend. |
| `elemArr` | integer | Yes | >= 0 | Element array level: `1` for non-stick dimensions, `2` for stick dimensions, or `0` when collapsed. Encodes whether this dimension maps to the innermost leaf elements or multiple sticks per slice. |
| `padding` | string | Yes | `"nopad"`, `"lowered_padded"`, `"padded_nozeropad"`, `"padded_wzeropad"`, `"padded_fullspan"`, or `"padded_fullspan_wunneeded"` | Padding state for this dimension in the allocated buffer. `"nopad"` = no padding; `"lowered_padded"` = padding collapsed into a lowered layout; `"padded_nozeropad"` = padded but region is not zeroed (used with conv2d when padding is non-zero); `"padded_wzeropad"` = padded and region is zero-filled; `"padded_fullspan"` = full-span padding; `"padded_fullspan_wunneeded"` = full-span padding with unneeded pad elements (used with conv2d when padding is zero). |
| `folds` | [FoldManager](foldmanager.md) | Yes | — | Defines how the coordinate value for this dimension is computed across the hierarchy using affine or constant fold functions. See [FoldProperty](foldproperty.md#folds-hierarchy) for hierarchy labels. |

## Example

A `CoordinateInfo` object for dimension `"mb"` as it appears inside `coordInfo`:

```json
"coordInfo": {
  "mb": {
    "spatial":  0,
    "temporal": 0,
    "elemArr":  0,
    "padding":  "nopad",
    "folds": {
      "dim_prop_func": [{"Const": {}}],
      "dim_prop_attr": [{"factor_": 32, "label_": "mb"}],
      "data_": {
        "[0]": 0
      }
    }
  }
}
```

In this example, `"mb"` is at spatial level 0, temporal level 0, element-array level 0 — all
collapsed to a single level. This is a simplified illustration; in a fully tiled DSC the same
dimension would typically have `spatial: 3` (core, corelet, row split levels) and `elemArr: 1`
(non-stick) or `2` (stick), as shown in the [`ScheduleTreeNode`](scheduletreenode.md) worked
examples. Use `spatial: 0` / `elemArr: 0` only when the dimension is not partitioned across
the spatial hierarchy.

---

| [← Previous: CoordinateContainer](coordinatecontainer.md) | [↑ Table of Contents](README.md) | [Next: ComputeOperation →](computeoperation.md) |
|:--|:--:|--:|
