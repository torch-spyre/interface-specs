# ConstantInfo

Constant information.

**Written by**: `generate_constant_info()`

## Required Fields

- `name_`
- `dataFormat_`
- `data_`

## Structure

```json
{
  "name_": "<string>",
  "dataFormat_": "<format>",
  "data_": <FoldManager>
}
```

## Field Descriptions

### name_
- **Type**: `string`
- **Description**: Constant name
- **Required**: Yes

### dataFormat_
- **Type**: `string`
- **Description**: Data format
- **Required**: Yes
- **Enum**: `["SEN169_FP16", "SEN169_BFP16", "SEN169_FP32", "SEN169_INT8", "SEN169_INT16", "SEN169_INT32"]`

### data_
- **Type**: `FoldManager`
- **Description**: Constant data using FoldManager
- **Required**: Yes

## Example

```json
{
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

[← Back to Table of Contents](README.md)
