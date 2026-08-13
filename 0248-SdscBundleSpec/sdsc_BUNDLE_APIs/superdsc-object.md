# SuperDsc Object

**Path**: `<operation_name>`

Bundle-level configuration for multi-core accelerator operation..

## Required Fields

- `coreFoldProp_`
- `coreletFoldProp_`
- `numCoresUsed_`
- `coreIdToDsc_`
- `coreIdToDscSchedule`
- `dscs_`

## Structure

```json
{
  "sdscFoldProps_": [<FoldProperty>, ...],
  "sdscFolds_": <FoldManager>,
  "coreFoldProp_": <FoldProperty>,
  "coreletFoldProp_": <FoldProperty>,
  "numCoresUsed_": <int>,
  "coreIdToDsc_": <map<string, int>>,
  "numWkSlicesPerDim_": <map<string, int>>,
  "coreIdToWkSlice_": <map<string, map<string, int>>>,
  "coreIdToDscSchedule": <map<string, array<array<int>>>>,
  "dscs_": [<WrappedDesignSpaceConfig>, ...]
}
```

## Field Descriptions

### sdscFoldProps_
- **Type**: `array of FoldProperty`
- **Description**: SDSC-level fold properties
- **Optional**: Yes

### sdscFolds_
- **Type**: `FoldManager`
- **Description**: SDSC-level fold manager
- **Optional**: Yes

### coreFoldProp_
- **Type**: `FoldProperty`
- **Description**: Core fold properties
- **Required**: Yes

### coreletFoldProp_
- **Type**: `FoldProperty`
- **Description**: Corelet fold properties
- **Required**: Yes

### numCoresUsed_
- **Type**: `integer` (minimum: 1)
- **Description**: Total number of cores used
- **Required**: Yes

### coreIdToDsc_
- **Type**: `map<string, integer>`
- **Description**: Mapping from core ID (as string) to DSC index
- **Required**: Yes
- **Key Pattern**: `^[0-9]+$`
- **Value**: Integer >= 0

**Example**:
```json
{
  "coreIdToDsc_": {
    "0": 0,
    "1": 0
  }
}
```

### numWkSlicesPerDim_
- **Type**: `map<string, integer>`
- **Description**: Number of work slices per dimension
- **Optional**: Yes
- **Key Pattern**: `^[A-Za-z_][A-Za-z0-9_]*$`
- **Value**: Integer >= 1

**Example**:
```json
{
  "numWkSlicesPerDim_": {
    "M": 2,
    "N": 1,
    "K": 1
  }
}
```

### coreIdToWkSlice_
- **Type**: `map<string, map<string, integer>>`
- **Description**: Mapping from core ID to work slice indices per dimension
- **Optional**: Yes
- **Outer Key Pattern**: `^[0-9]+$` (core ID)
- **Inner Key Pattern**: `^[A-Za-z_][A-Za-z0-9_]*$` (dimension name)
- **Value**: Integer >= 0

**Example**:
```json
{
  "coreIdToWkSlice_": {
    "0": {
      "M": 0,
      "N": 0,
      "K": 0
    },
    "1": {
      "M": 1,
      "N": 0,
      "K": 0
    }
  }
}
```

### coreIdToDscSchedule
- **Type**: `map<string, array<array<integer>>>`
- **Description**: Per-core DSC scheduling information
- **Required**: Yes
- **Key Pattern**: `^[0-9]+$` (core ID)
- **Value**: Array of 4-element arrays `[datadsc_idx, dldsc_idx, before_sync, after_sync]`

**Example**:
```json
{
  "coreIdToDscSchedule": {
    "0": [[0, 0, 0, 0]],
    "1": [[0, 0, 0, 0]]
  }
}
```

### dscs_
- **Type**: `array of WrappedDesignSpaceConfig`
- **Description**: Array of Design Space Configurations
- **Required**: Yes
- **Min Items**: 1

---

[← Back to Table of Contents](README.md)
