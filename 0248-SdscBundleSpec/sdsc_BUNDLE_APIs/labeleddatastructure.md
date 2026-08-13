# LabeledDataStructure

Labeled data structure definition.

## Required Fields

- `ldsIdx_`
- `dsName_`
- `dsType_`
- `dataFormat_`
- `memOrg_`

## Structure

```json
{
  "ldsIdx_": <int>,
  "dsName_": "<string>",
  "dsType_": "INPUT" | "OUTPUT" | "KERNEL" | "KERNEL_IDX",
  "scale_": [<number>, ...],
  "wordLength": <int>,
  "dataFormat_": "<format>",
  "memOrg_": <MemoryOrganization>
}
```

## Field Descriptions

### ldsIdx_
- **Type**: `integer` (>= 0)
- **Description**: Labeled data structure index
- **Required**: Yes

### dsName_
- **Type**: `string`
- **Description**: Data structure name
- **Required**: Yes

### dsType_
- **Type**: `string`
- **Description**: Data structure type (from torch_spyre LAYOUT_INPUT_LABELS and LAYOUT_OUTPUT_LABELS)
- **Required**: Yes
- **Enum**: `["INPUT", "OUTPUT", "KERNEL", "KERNEL_IDX"]`

### scale_
- **Type**: `array of number`
- **Description**: Scale factors
- **Optional**: Yes

### wordLength
- **Type**: `integer` (>= 1)
- **Description**: Word length in bytes
- **Optional**: Yes

### dataFormat_
- **Type**: `string`
- **Description**: Data format
- **Required**: Yes
- **Enum**: `["SEN169_FP16", "SEN169_BFP16", "SEN169_FP32", "SEN169_INT8", "SEN169_INT16", "SEN169_INT32"]`

### memOrg_
- **Type**: `MemoryOrganization`
- **Description**: Memory organization
- **Required**: Yes

## Example

```json
{
  "ldsIdx_": 0,
  "dsName_": "input",
  "dsType_": "INPUT",
  "scale_": [1.0, 1.0, 1.0],
  "wordLength": 2,
  "dataFormat_": "SEN169_FP16",
  "memOrg_": {
    "hbm": {"isPresent": 1},
    "lx": {"isPresent": 0}
  }
}
```

---

[← Back to Table of Contents](README.md)
