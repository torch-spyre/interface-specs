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
  "padding":  "nopad" | "pad",
  "folds":    <FoldManager>
}
```

## Fields

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `spatial` | integer | Yes | >= 0 | Spatial memory hierarchy level. Encodes how this dimension is distributed across cores (the spatial dimension of the Spyre memory model). |
| `temporal` | integer | Yes | >= 0 | Temporal memory hierarchy level. Encodes how this dimension maps to time steps / LX scratchpad staging passes. |
| `elemArr` | integer | Yes | >= 0 | Element array level. Encodes the position of this dimension within a stick (the innermost element-array dimension). |
| `padding` | string | Yes | `"nopad"` or `"pad"` | Whether this dimension is padded in the allocated buffer. Use `"pad"` when the dimension has padding applied (see [Padding](padding.md)); `"nopad"` otherwise. |
| `folds` | [FoldManager](foldmanager.md) | Yes | — | Defines how the coordinate value for this dimension is computed per core and time step using affine or constant fold functions. |

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

In this example, `"mb"` is at spatial level 0, temporal level 0, element-array level 0 (all collapsed
to a single level), has no padding, and uses a constant fold function with a factor of 32.

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: LabeledDataStructure →](labeleddatastructure.md) |
|:--|:--:|--:|
