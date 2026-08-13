# Root Structure

The root of an SDSC Bundle JSON file uses a **pattern property** where the key is the operation name.

## Structure

```json
{
  "<operation_name>": {
    "$ref": "#/$defs/SuperDsc"
  }
}
```

## Pattern

- **Key Pattern**: `^[a-zA-Z0-9_/\\-][a-zA-Z0-9_/\\-]*$`
- **Value**: SuperDsc object
- **Properties**: Exactly 1 property (the operation name)

## Example

```json
{
  "gelu_forward": {
    "coreFoldProp_": {...},
    "coreletFoldProp_": {...},
    "numCoresUsed_": 2,
    "coreIdToDsc_": {...},
    "coreIdToDscSchedule": {...},
    "dscs_": [...]
  }
}
```

---

[← Back to Table of Contents](README.md)
