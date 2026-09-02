# SuperDsc Object

`SuperDsc` is the top-level object of every SDSC JSON file. It orchestrates
multi-core execution by defining how work is distributed across Spyre cores:
fold properties describe how each core's data address is computed, core maps
assign each core to a `DesignSpaceConfig`, and the per-core schedule encodes
when each DSC runs on each core. The `dscs_` array holds the actual
per-operation compute configurations.

## Context

A `SuperDsc` object is the value of the single top-level key in an SDSC JSON
file. That key is the operation name and must match the pattern
`^[a-zA-Z0-9_/\-][a-zA-Z0-9_/\-]*$`. Each JSON file contains exactly one
such key (`minProperties: 1`, `additionalProperties: false`).

```json
{
  "gelu_forward": {
    "coreFoldProp_": { ... },
    "coreletFoldProp_": { ... },
    "numCoresUsed_": 2,
    "coreIdToDsc_": { "0": 0, "1": 0 },
    "coreIdToDscSchedule": { ... },
    "dscs_": [ ... ]
  }
}
```

Each entry in `dscs_` is a single-key object — the key is the operation name
(same pattern as the bundle root key) and the value is a full
[`DesignSpaceConfig`](designspaceconfig.md).

## Structure

```json
{
  "sdscFoldProps_":                        [<FoldProperty>, ...],
  "sdscFolds_":                            <FoldManager>,
  "coreFoldProp_":                         <FoldProperty>,
  "coreletFoldProp_":                      <FoldProperty>,
  "numCoresUsed_":                         <int>,
  "dimToSymbolMappingOpcodeCorrection_":   <map<string, string>>,
  "inputSymbolsAndTags_":                  <map<string, string>>,
  "symbolDefinitions_":                    <object>,
  "coreIdToDsc_":                          <map<string, int>>,
  "numWkSlicesPerDim_":                    <map<string, int>>,
  "coreIdToWkSlice_":                      <map<string, map<string, int>>>,
  "coreIdToDscSchedule":                   <map<string, array<array<int>>>>,
  "dscs_":                                 [<WrappedDesignSpaceConfig>, ...]
}
```

## Fields

Six fields are required. No additional properties are allowed.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `sdscFoldProps_` | array of [FoldProperty](foldproperty.md) | No | — | SDSC-level fold properties when the bundle spans multiple fold dimensions above the core level. |
| `sdscFolds_` | [FoldManager](foldmanager.md) | No | — | Fold manager encoding addresses or mappings at the bundle level, above the per-core level. |
| `coreFoldProp_` | [FoldProperty](foldproperty.md) | Yes | — | Fold factor and label for the core level of the memory hierarchy (e.g. `factor_: 2, label_: "core"`). |
| `coreletFoldProp_` | [FoldProperty](foldproperty.md) | Yes | — | Fold factor and label for the corelet level of the memory hierarchy (e.g. `factor_: 2, label_: "corelet"`). |
| `numCoresUsed_` | integer | Yes | >= 1 | Total number of Spyre cores used across all DSCs in this bundle. |
| `dimToSymbolMappingOpcodeCorrection_` | map&lt;string, string&gt; | No | Keys: dim names | Symbol mapping corrections applied during opcode generation. Keys are dimension names; values are corrected symbol names. |
| `inputSymbolsAndTags_` | map&lt;string, string&gt; | No | Keys: symbol names | Input symbols and their associated tags for symbolic dimension resolution. |
| `symbolDefinitions_` | object | No | — | Variable definitions for symbolic dimensions used across the bundle. |
| `coreIdToDsc_` | map&lt;string, integer&gt; | Yes | Keys: `^[0-9]+$`; values >= 0 | Maps each core ID (string integer) to a zero-based index into `dscs_`. Multiple cores with the same index share one `DesignSpaceConfig`. |
| `numWkSlicesPerDim_` | map&lt;string, integer&gt; | No | Keys: dim names; values >= 1 | Total number of work slices per dimension across all cores. |
| `coreIdToWkSlice_` | map&lt;string, map&lt;string, integer&gt;&gt; | No | Outer keys: core IDs; inner keys: dim names; values >= 0 | Maps each core ID to a map of dimension name → work slice index assigned to that core. |
| `coreIdToDscSchedule` | map&lt;string, array&lt;array&lt;int&gt;&gt;&gt; | Yes | Keys: `^[0-9]+$`; inner arrays: exactly 4 integers | Per-core execution schedule. Each inner array is a step tuple `[datadsc_idx, dldsc_idx, before_sync, after_sync]`: data DSC index, data-load DSC index, barrier before step (0 = none), barrier after step (0 = none). |
| `dscs_` | array of object | Yes | >= 1 item | Array of Design Space Configurations. Each entry is a single-key object `{"<op_name>": <DesignSpaceConfig>}` — see [`DesignSpaceConfig`](designspaceconfig.md). |

**Note on field naming:** `coreIdToDscSchedule` lacks the trailing underscore used by most other
fields. The serialized key in the JSON bundle is `coreIdToDscSchedule` (no underscore) — this
inconsistency is a known anomaly. Do not add a trailing underscore when writing bundle JSON.

## Example

A 2-core GELU operation where both cores share one `DesignSpaceConfig` (index `0`), with a
single schedule step per core.

```json
{
  "gelu_forward": {
    "coreFoldProp_":    {"factor_": 2, "label_": "core"},
    "coreletFoldProp_": {"factor_": 2, "label_": "corelet"},
    "numCoresUsed_": 2,
    "coreIdToDsc_": {
      "0": 0,
      "1": 0
    },
    "numWkSlicesPerDim_": {
      "mb": 2
    },
    "coreIdToWkSlice_": {
      "0": {"mb": 0},
      "1": {"mb": 1}
    },
    "coreIdToDscSchedule": {
      "0": [[0, 0, 0, 0]],
      "1": [[0, 0, 0, 0]]
    },
    "dscs_": [
      {
        "gelu": {
          "numCoresUsed_": 2,
          "coreIdsUsed_": [0, 1],
          "N_": {"mb_": 32, "in_": 128},
          "labeledDs_": [...],
          "scheduleTree_": [...],
          "computeOp_": [...]
        }
      }
    ]
  }
}
```

Both cores execute DSC index `0` (`"gelu"`). Core `0` processes work slice `mb=0` and core `1`
processes work slice `mb=1`. The schedule tuple `[0, 0, 0, 0]` means: execute data DSC 0,
data-load DSC 0, no barrier before, no barrier after.

---

| [← Previous: Object Hierarchy](JSON-object-Hierarchy.md) | [↑ Table of Contents](README.md) | [Next: FoldProperty →](foldproperty.md) |
|:--|:--:|--:|
