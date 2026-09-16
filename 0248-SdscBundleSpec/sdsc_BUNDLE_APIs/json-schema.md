# JSON Schema

[`sdscbundle-schema.json`](sdscbundle-schema.json) is the machine-readable contract for every
`sdsc_*.json` file in a SuperDSC-Bundle. It is written against
[JSON Schema draft 2020-12](https://json-schema.org/draft/2020-12) and is currently at
**version 1.2.0**.

The schema enforces:

- Required fields and their types at every level of the object hierarchy
- Enum values for `dsType_`, `component_`, `indirectAllocType_`, `indexTensorType_`,
  `dataFormat_`, `fidelity_`, `nodeType_`, and all other constrained string fields
- Pattern constraints on dynamic keys such as core IDs and operation-name root keys
- Structural rules such as `additionalProperties: false` on all defined objects

Semantic constraints that go beyond what JSON Schema can express — cross-field consistency,
stick-size alignment, and symbol-ID correspondence — are documented in the individual object
pages and in the [Validation section of SDSC JSON API](SDSC-json-api.md#validation).

## Version history

| Version | Date | Summary |
|---|---|---|
| 1.0.0 | — | Initial schema |
| 1.0.1 | 2026-05-11 | Corrected `dsType_` enum to `INPUT`, `OUTPUT`, `KERNEL`, `KERNEL_IDX` |
| 1.1.0 | 2026-05-XX | Added symbolic-dimension support: `SymbolicDimInfo`, `DimPaddingSizes`, `dimToSymbolMapping_`, `isStartAddrSymbolic_`, and related `SuperDsc` symbol fields |
| 1.1.1 | 2026-07-07 | Allow `isStartAddrSymbolic_` to accept integer `0`/`1` in addition to boolean |
| 1.1.2 | 2026-07-07 | Allow `DataStructDims` dimension fields to accept empty object `{}` |
| 1.2.0 | 2026-07-08 | Sync with deeptools-schema.json v1.3.0: explicit dim fields on `DataStructDims`, corrected `DimPaddingSizes` field names, corrected `SymbolicDimInfo.maxSize_`, added `name_` to per-stage `dataStageParam_` object |

---

| [← Previous: Object Hierarchy](JSON-object-Hierarchy.md) | [↑ Table of Contents](README.md) | [Next: SuperDsc Object →](superdsc-object.md) |
|:--|:--:|--:|
