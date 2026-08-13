# ScheduleTreeNode

Schedule tree node for memory allocation or data movement.

## Required Fields

- `nodeType_`
- `name_`

## Structure

```json
{
  "nodeType_": "allocate",
  "name_": "<string>",
  "prev_": "<string>",
  "ldsIdx_": <int>,
  "component_": "hbm" | "lx",
  "layoutDimOrder_": [<string>, ...],
  "maxDimSizes_": [<int>, ...],
  "startAddressCoreCorelet_": <FoldManager>,
  "backGapCore_": <map<string, map<string, string>>>,
  "coordinates_": <CoordinateContainer>
}
```

## Field Descriptions

### nodeType_
- **Type**: `string`
- **Description**: Node type
- **Required**: Yes
- **Enum**: `["allocate"]`

### name_
- **Type**: `string`
- **Description**: Node name
- **Required**: Yes

### prev_
- **Type**: `string`
- **Description**: Previous node name
- **Optional**: Yes

### ldsIdx_
- **Type**: `integer` (>= 0)
- **Description**: Labeled data structure index
- **Optional**: Yes

### component_
- **Type**: `string`
- **Description**: Memory component
- **Optional**: Yes
- **Enum**: `["hbm", "lx"]`

### layoutDimOrder_
- **Type**: `array of string`
- **Description**: Layout dimension order
- **Optional**: Yes

### maxDimSizes_
- **Type**: `array of integer`
- **Description**: Maximum dimension sizes
- **Optional**: Yes

### startAddressCoreCorelet_
- **Type**: `FoldManager`
- **Description**: Start address per core/corelet
- **Optional**: Yes

### backGapCore_
- **Type**: `map<string, map<string, string>>`
- **Description**: Back gaps in number of elements per dimension per core
- **Written conditionally**: If `tensor.backGap` is True
- **Optional**: Yes
- **Outer Key Pattern**: `^[A-Za-z_][A-Za-z0-9_]*$` (dimension name)
- **Inner Key Pattern**: `^-?[0-9]+$` (core ID)
- **Value**: String representation of gap size

### coordinates_
- **Type**: `CoordinateContainer`
- **Description**: Coordinate information
- **Optional**: Yes

## Example

```json
{
  "nodeType_": "allocate",
  "name_": "input_alloc",
  "ldsIdx_": 0,
  "component_": "hbm",
  "layoutDimOrder_": ["mb", "out", "in"],
  "maxDimSizes_": [32, 128, 64],
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
}
```

---

[← Back to Table of Contents](README.md)
