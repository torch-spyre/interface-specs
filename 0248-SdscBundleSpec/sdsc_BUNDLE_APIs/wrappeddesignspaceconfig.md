# WrappedDesignSpaceConfig

DSC wrapped in operation name key — used in `dscs_` array.

## Structure

```json
{
  "<operation_name>": <DesignSpaceConfig>
}
```

- **Key Pattern**: `^[a-zA-Z0-9_/\\-][a-zA-Z0-9_/\\-]*$`
- **Properties**: Exactly 1 property

## Example

```json
{
  "gelu": {
    "numCoresUsed_": 2,
    "coreIdsUsed_": [0, 1],
    "N_": {...},
    "computeOp_": [...],
    "labeledDs_": [...],
    "scheduleTree_": [...]
  }
}
```

---

[← Back to Table of Contents](README.md)
