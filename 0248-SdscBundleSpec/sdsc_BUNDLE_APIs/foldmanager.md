# FoldManager

Fold manager for multi-dimensional data access.

## Structure

```json
{
  "dim_prop_func": [<DimPropFunc>, ...],
  "dim_prop_attr": [<FoldProperty>, ...],
  "data_": {
    "[index1, index2, ...]": <value>
  }
}
```

## Fields

### dim_prop_func
- **Type**: `array of objects`
- **Description**: Dimension property functions
- **Required**: Yes
- **Possible Values**: `Const`, `Map`, `Affine`, `WkSplit`

**Const**:
```json
{"Const": {}}
```

**Map**:
```json
{"Map": {}}
```

**Affine**:
```json
{
  "Affine": {
    "alpha_": <int>,
    "beta_": <int>
  }
}
```

**WkSplit**:
```json
{"WkSplit": {}}
```

### dim_prop_attr
- **Type**: `array of FoldProperty`
- **Description**: Dimension property attributes
- **Required**: Yes

### data_
- **Type**: `object`
- **Description**: Data values indexed by fold coordinates
- **Optional**: Yes
- **Key Pattern**: `^\\[[0-9, ]+\\]$` (e.g., "[0, 0]", "[1, 0]")
- **Value**: String, integer, or array of integers

## Example

```json
{
  "dim_prop_func": [
    {"Map": {}},
    {"Const": {}}
  ],
  "dim_prop_attr": [
    {"factor_": 2, "label_": "core"},
    {"factor_": 2, "label_": "corelet"}
  ],
  "data_": {
    "[0, 0]": "1024",
    "[1, 0]": "2048"
  }
}
```

---

[← Back to Table of Contents](README.md)
