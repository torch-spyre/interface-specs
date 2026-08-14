# ConstantInfo

A `ConstantInfo` entry defines a named, typed constant value (e.g. a scaling
factor or clip bound) used by a `DesignSpaceConfig` operation. Because
constants can vary per-core or per-time-step, the value is encoded as a
[`FoldManager`](foldmanager.md) rather than a plain scalar — the fold
structure distributes the right value to each core and time step at runtime.

## Context

`ConstantInfo` entries appear as values of the `constantInfo_` field inside
[`DesignSpaceConfig`](designspaceconfig.md), keyed by string integer (`"0"`,
`"1"`, …). The entire field may also be the literal string `"{}"` when the
operation has no constants.

```json
"constantInfo_": {
  "0": { <ConstantInfo> },
  "1": { <ConstantInfo> }
}

// or, when no constants:
"constantInfo_": "{}"
```

## Structure

```json
{
  "name_":       "<string>",
  "dataFormat_": "<format>",
  "data_":       <FoldManager>
}
```

## Fields

All three fields are required. No additional properties are allowed.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `name_` | string | Yes | — | Unique name for this constant within the DSC (e.g. `"scaling_factor"`, `"eps"`). Referenced by `opFuncName` operations that consume it, such as `"mean"` or `"layernormscale"`. |
| `dataFormat_` | string enum | Yes | see [Data Formats](#data-formats) | Numeric format of the constant's elements. Must match the data format of the tensors the constant is applied to. |
| `data_` | [FoldManager](foldmanager.md) | Yes | — | The constant value(s), encoded as a fold structure so different values can be assigned per-core or per-time-step. |

## Data Formats

The schema enforces the following Spyre-validated subset. The full list of formats
supported by the torch-spyre runtime is documented in
[SDSC-json-api.md](SDSC-json-api.md).

| Value | Description |
|---|---|
| `SEN169_FP16` | SEN169 16-bit floating point (most common for Spyre) |
| `SEN169_BFP16` | SEN169 brain float 16 |
| `SEN169_FP32` | SEN169 32-bit floating point |
| `SEN169_INT8` | SEN169 8-bit integer |
| `SEN169_INT16` | SEN169 16-bit integer |
| `SEN169_INT32` | SEN169 32-bit integer |

## Example

The example below shows a `scaling_factor` constant for a mean reduction
operation. The constant is uniform across all cores and time steps (`Const`
functions at both fold levels). The value `15872` is the FP16 bit-pattern
for `0.25` (used as `1/N` scaling). The fold coordinate `[0, 0]` addresses
core index 0, time-step index 0.

```json
"constantInfo_": {
  "0": {
    "name_": "scaling_factor",
    "dataFormat_": "SEN169_FP16",
    "data_": {
      "dim_prop_func": [{"Const": {}}, {"Const": {}}],
      "dim_prop_attr": [
        {"factor_": 2, "label_": "core"},
        {"factor_": 2, "label_": "corelet"}
      ],
      "data_": {
        "[0, 0]": [15872]
      }
    }
  }
}
```

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: Complete Example (JSON) →](complete-example.md) |
|:--|:--:|--:|
