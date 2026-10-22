# SuperDSC Object

`SuperDsc` is the top-level object of every SDSC JSON file. It orchestrates
multi-core execution by defining how work is distributed across Spyre cores:
work-slice mappings partition the iteration space, and the `dscs_` array holds
the compute configuration that every core runs.

## Context

A `SuperDsc` object is the value of the single top-level key in an SDSC JSON
file. That key is the operation name and MUST match the pattern
`^[a-zA-Z0-9_/\-][a-zA-Z0-9_/\-]*$`. Each JSON file contains exactly one
such key (`minProperties: 1`, `additionalProperties: false`).

```json
{
  "gelu_forward": {
    "numCoresUsed_": 2,
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
  "numCoresUsed_":                         <int>,
  "inputSymbolsAndTags_":                  <map<string, string>>,
  "numWkSlicesPerDim_":                    <map<string, int>>,
  "coreIdToWkSlice_":                      <map<string, map<string, int>>>,
  "dscs_":                                 [<WrappedDesignSpaceConfig>]
}
```

## Fields

Two fields are required. No additional properties are allowed.

| Field | Type | Required † | Constraints | Description |
|---|---|---|---|---|
| `numCoresUsed_` | integer | Yes | >= 1 | Total number of Spyre cores used across all DSCs in this bundle. |
| `debug_handle_` | DebugHandle or null | No† | — | Source-to-kernel provenance for this SuperDSC. Consumed by tooling that maps a compiled kernel back to its source; it does not affect compilation. Includes the source file/line (`source`), ATen op name (`aten_op`), lowering chain (`ir_chain`), fusion origins (`fused_from`), and rewrite history (`transform_history`). `null` is a normal value (not missing data) when provenance is unavailable. |
| `inputSymbolsAndTags_` | map&lt;string, string&gt; | No | Keys: symbol names | Input symbols and their associated tags for symbolic dimension resolution. |
| `numWkSlicesPerDim_` | map&lt;string, integer&gt; | No | Keys: dim names; values >= 1 | Total number of work slices per dimension across all cores. |
| `coreIdToWkSlice_` | map&lt;string, map&lt;string, integer&gt;&gt; | No | Outer keys: core IDs; inner keys: dim names; values >= 0 | Maps each core ID to a map of dimension name → work slice index assigned to that core. |
| `dscs_` | array of object | Yes | exactly 1 item | The bundle's Design Space Configuration, as a one-element array. The entry is a single-key object `{"<op_name>": <DesignSpaceConfig>}` — see [`DesignSpaceConfig`](designspaceconfig.md). Every core listed in its `coreIdsUsed_` runs it. |

**† `debug_handle_` presence convention:** `debug_handle_` is optional. When
provenance information is unavailable, producers MAY either omit the field or
set its value to `null`.

## Example

A 2-core GELU operation where both cores share one `DesignSpaceConfig`.

```json
{
  "gelu_forward": {
    "numCoresUsed_": 2,
    "numWkSlicesPerDim_": {
      "mb": 2
    },
    "coreIdToWkSlice_": {
      "0": {"mb": 0},
      "1": {"mb": 1}
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

Both cores process their respective work slices (`mb=0` and `mb=1`).

---

[↑ Spec Map](../../SuperDSC-Bundle.md#spec-map)|
