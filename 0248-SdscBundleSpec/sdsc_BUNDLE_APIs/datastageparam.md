# DataStageParam

Data staging parameters.

## Structure

```json
{
  "ss_": <DataStructDims>,
  "el_": <DataStructDims>
}
```

## Fields

- **`ss_`**: Stage size dimensions (steady-state)
- **`el_`**: Element dimensions (epilogue)

## Example

```json
{
  "0": {
    "ss_": {
      "mb_": 16,
      "out_": 64
    },
    "el_": {
      "mb_": 8,
      "out_": 32
    }
  }
}
```

---

[← Back to Table of Contents](README.md)
