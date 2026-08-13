# PrimaryDsInfo

Primary data structure layout information.

## Structure

```json
{
  "layoutDimOrder_": [<string>, ...],
  "stickDimOrder_": [<string>, ...],
  "stickSize_": [<int>, ...]
}
```

## Fields

- **`layoutDimOrder_`**: (optional) Layout dimension order
- **`stickDimOrder_`**: (optional) Stick dimension order
- **`stickSize_`**: (optional) Stick sizes (each integer >= 1)

## Example

```json
{
  "INPUT": {
    "layoutDimOrder_": ["mb", "out", "in"],
    "stickDimOrder_": ["in"],
    "stickSize_": [64]
  }
}
```

---

[← Back to Table of Contents](README.md)
