# CoordinateContainer

Container for per-dimension coordinate information within a `ScheduleTreeNode`'s `coordinates_` field.
It groups the coordinate info for all dimensions of a tensor alongside the core-to-work-slice mapping.

## Context

`CoordinateContainer` appears as the value of the `coordinates_` field inside each `ScheduleTreeNode`:

```json
"scheduleTree_": [
  {
    "nodeType_": "allocate",
    "coordinates_": { <CoordinateContainer> }
  }
]
```

## Structure

```json
{
  "coordInfo": {
    "<dim_name>": <CoordinateInfo>
  },
  "coreIdToWkSlice_": {
    "<core_id>": {
      "<dim_name>": <slice_index>
    }
  }
}
```

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `coordInfo` | `map<string, CoordinateInfo>` | No | Per-dimension coordinate information. Keys are dimension names (pattern `^[a-z_][a-z0-9_]*$`, e.g. `mb`, `kb`). See [CoordinateInfo](coordinateinfo.md). |
| `coreIdToWkSlice_` | `object` | No | Maps each core ID (as a string integer key) to a map of dimension name → work slice index assigned to that core for that dimension. |

## Example

```json
"coordinates_": {
  "coordInfo": {
    "mb": {
      "spatial": 0,
      "temporal": 0,
      "elemArr": 0,
      "padding": "nopad",
      "folds": {
        "dim_prop_func": [{"Const": {}}],
        "dim_prop_attr": [{"factor_": 32, "label_": "mb"}],
        "data_": { "[0]": 0 }
      }
    }
  },
  "coreIdToWkSlice_": {
    "0":  { "mb": 0 },
    "1":  { "mb": 1 },
    "2":  { "mb": 2 }
  }
}
```

---

| [← Previous: ScheduleTreeNode](scheduletreenode.md) | [↑ Table of Contents](README.md) | [Next: CoordinateInfo →](coordinateinfo.md) |
|:--|:--:|--:|
