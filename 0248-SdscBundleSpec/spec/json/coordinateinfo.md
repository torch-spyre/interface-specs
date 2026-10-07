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
| `spatial` | integer | Yes | `3` | The frontend always emits `3`: the dimension is partitioned across cores. |
| `temporal` | integer | Yes | `0` | Always set to `0` by the frontend. |
| `elemArr` | integer | Yes | >= 0 | Element array level: `1` for non-stick dimensions, `2` for stick dimensions, or `0` when collapsed. Encodes whether this dimension maps to the innermost leaf elements or multiple sticks per slice. |
| `padding` | string | Yes | `"nopad"`, `"lowered_padded"`, `"padded_nozeropad"`, `"padded_wzeropad"`, `"padded_fullspan"`, or `"padded_fullspan_wunneeded"` | Padding state for this dimension in the allocated buffer. `"nopad"` = no padding; `"lowered_padded"` = padding collapsed into a lowered layout; `"padded_nozeropad"` = padded but region is not zeroed (used with conv2d when padding is non-zero); `"padded_wzeropad"` = padded and region is zero-filled; `"padded_fullspan"` = full-span padding; `"padded_fullspan_wunneeded"` = full-span padding with unneeded pad elements (used with conv2d when padding is zero). |
| `folds` | [FoldManager](foldmanager.md) | Yes | — | Defines how the coordinate value for this dimension is computed across the hierarchy using affine or constant fold functions. See [FoldProperty](foldproperty.md#folds-hierarchy) for hierarchy labels. |

## Example

A `CoordinateInfo` object for dimension `"mb"` as it appears inside `coordInfo`:

```json
"coordInfo": {
  "mb": {
    "spatial":  3,
    "temporal": 0,
    "elemArr":  1,
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

In this example, `"mb"` is partitioned across the spatial hierarchy (`spatial: 3`, covering the
core, corelet, and row-split levels), has no temporal partitioning (`temporal: 0`), and maps to
non-stick elements (`elemArr: 1`). See the [`ScheduleTreeNode`](scheduletreenode.md) worked
examples for further detail.

---

[↑ Spec Map](../../SuperDSC-Bundle.md#spec-map)|
