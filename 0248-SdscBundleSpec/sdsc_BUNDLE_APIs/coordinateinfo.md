# CoordinateInfo

Coordinate information for a dimension.

## Required Fields

- `spatial`
- `temporal`
- `elemArr`
- `padding`
- `folds`

## Structure

```json
{
  "spatial": <int>,
  "temporal": <int>,
  "elemArr": <int>,
  "padding": "nopad" | "pad",
  "folds": <FoldManager>
}
```

## Field Descriptions

- **`spatial`**: Spatial level (integer >= 0)
- **`temporal`**: Temporal level (integer >= 0)
- **`elemArr`**: Element array level (integer >= 0)
- **`padding`**: Padding type ("nopad" or "pad")
- **`folds`**: Fold manager for this coordinate

## Example

```json
{
  "mb": {
    "spatial": 0,
    "temporal": 0,
    "elemArr": 0,
    "padding": "nopad",
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

---

[← Back to Table of Contents](README.md)
