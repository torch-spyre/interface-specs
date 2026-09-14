# SDSC Bundle API — Change Log

This file records changes to the API documentation after each task or Bob session.
It is the working log that accumulates until a PR is merged.

**On PR merge:** review all entries since the last version tag, determine the
SemVer bump (MAJOR / MINOR / PATCH), update `Spec Version` and `Git Baseline`
in [README.md](README.md), and add a one-line summary entry to the
[Released Versions](#released-versions) table below.

> **How to add an entry during a task:**
> Prepend a new `### Task` block at the top of [Unreleased Changes](#unreleased-changes).
> Include the date, a brief description of what the task was, the files changed,
> and the commits involved.

---

## Unreleased Changes

### Task — Fix opFuncName and labeled-DS tensor references in complete-example.md (gap analysis items 9 & 10)

| File | Change |
|---|---|
| `complete-example.md` | Fix `opFuncName` value from `"gelu"` to `"gelufwd"` (item 9 — correct OpFunc string for `GELU_FWD` as defined in `SuperDSC-Bundle.md` and `computeoperation.md`). Fix `inputLabeledDs` and `outputLabeledDs` tensor reference strings from bare `"input"` / `"output"` to `"input-idx0"` / `"output-idx1"` to match the required `"<dsName_>-idx<ldsIdx_>"` form (item 10). |

---

### Task — Fix dim_prop_func bare-string form (gap analysis item 8)

| File | Change |
|---|---|
| `MLIR-bundle-usage-examples.md` | Replace 6 bare-string `dim_prop_func` entries (`["Const"]`, `["Map", "Const"]`) with the normative single-key object form (`[{"Const": {}}]`, `[{"Map": {}}, {"Const": {}}]`) as defined in `foldmanager.md`. All occurrences were inside `startAddressCoreCorelet_` blocks. No other file in the directory was affected. |

---

### Task — Resolve scf.for symbolic loop-bound contradiction (2026-09-09)

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Replace contradictory `scf.for` constraint (which simultaneously allowed and denied symbolic upper bounds) with the normative statement: all three bounds must be compile-time constants, no symbolic or runtime bounds are supported in any loop kind. Remove stale "loop upper bounds may also be symbolic" cross-reference from `arith.constant` description. |

---

### Decision — Per-document version tags deferred until formal release (2026-09-09)

Gap analysis item 14 (no per-document version tag linking to spec revision) closed as a deliberate design decision. Version tags will not be added to individual API documents until a formal release of the document set is made. The existing `Spec Version` and `Git Baseline` in `README.md` remain the single version anchor.

---

### Task — Document pdsRelation_ in designspaceconfig.md (2026-09-09)

| File | Change |
|---|---|
| `designspaceconfig.md` | Add `pdsRelation_` to Structure block and Fields table: `isPdsReuse` sub-field, in-place/separate buffer semantics, default behaviour when absent, emission condition (pool/window ops only) |

---

### Task — Add OpFuncs table to computeoperation.md; update cross-references (2026-09-09)

| File | Change |
|---|---|
| `computeoperation.md` | Replace external torch-spyre URL in "Supported Operations" with the full 80-row OpFuncs table sourced from SuperDSC-Bundle.md; add cross-link to stick-layout-constraints.md for ReStickifyOpHBM |
| `SDSC-json-api.md` | Update Step 9 `opFuncName` reference from external GitHub URL to `computeoperation.md#supported-operations` |
| `sdscbundle-schema.json` | Update `opFuncName` description from external URL to `computeoperation.md#supported-operations` |
| `Overview.md` | Change "full list of supported operations" to "full list of supported PyTorch operations" |

---

### Task — Document datadscs_ and add ReStickifyOpHBM stick constraints (2026-09-09)

| File | Change |
|---|---|
| `superdsc-object.md` | Add `datadscs_` to Structure block and Fields table; add `datadsc_idx` cross-reference in `coreIdToDscSchedule` description |
| `JSON-object-Hierarchy.md` | Add inline annotation to `datadscs_` entry explaining its role and `datadsc_idx` indexing relationship |
| `stick-layout-constraints.md` | Add "Stick Altering Data Shuffle" section for `ReStickifyOpHBM`: use-case (reshape/re-tiling, VirtualReshape), HBM-vs-LX path distinction, single-dimension input/output stick constraints, DF16 precision |

---

### Task — Add sdsc.json filling guide to SDSC-json-api.md (2026-09-09)

| File | Change |
|---|---|
| `SDSC-json-api.md` | Add "Filling a JSON File — Step-by-Step" section: 9-step guide ordered to match the `SuperDsc` object hierarchy; step-summary table mapping each step to its schema field and hierarchy level; all steps cross-link to reference docs; captures two filling rules from the spec that were absent from the API docs (`dataStageParam_` key/name convention, `startAddressCoreCorelet_` Map+Const default pattern) |

---

### Task — Specify SymbolicDimInfo object; cross-reference JSON ↔ MLIR names (2026-09-09)

| File | Change |
|---|---|
| `datastructdims.md` | Replace one-liner `symbolicDimInfo_` row with a full `SymbolicDimInfo` sub-section: fields table for `maxSize_` and `granularity_`, JSON ↔ MLIR name mapping table, and paired JSON + MLIR example |
| `MLIR-bundle-API.md` | Add explicit cross-references from `granularity` and `max_value` field descriptions in `sdscbundle.input_arg_extract` to `SymbolicDimInfo` in `datastructdims.md` |

---

### Task — Document granularity and max_value extraction in sdscbundle.input_arg_extract (2026-09-09)

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Expand `sdscbundle.input_arg_extract` to document `granularity` and `max_value` as extractable fields alongside `value`; add extended type syntax `!sdscbundle.input_arg<index, granularity=N, max_value=N>`; add new example for symbolic dimension extraction; update operations summary table and Bundle Container parameter table for consistency |

---

### Task — Fix confusing scf.for loop-carried variable constraint wording (2026-09-09)

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Merge two misleading bullets into one: "Loop-carried variables are not supported" and "Only the induction variable may be used directly inside the loop body" were confusing because the second implied other ops were banned. Replaced with a single bullet that states the constraint and clarifies the induction variable is freely usable in the loop body. |

---

## Released Versions

| Version | Date | Git Baseline | Summary |
|---|---|---|---|
| 1.0.0 | 2026-09-09 | `fcd337a` | Initial versioned baseline. All API docs brought to standard. See [v1.0.0 detail](#v100--2026-09-09) below. |

---

## v1.0.0 — 2026-09-09

Baseline version established from the full git history of the directory.
All commits prior to this version are recorded below, grouped by the task
that produced them.

---

### Task — Compliance review and schema/doc fixes (2026-09-09)

**Commits:** `fcd337a`, `06ae8f1`, `eb383ec`, `e816731`, `1a74153`, `ebeeb23`, `59ca249`

| File | Change |
|---|---|
| `padding.md` | Fix padding enum casing/field; consolidate schema files |
| `padding.md` | Restore original general intro paragraph |
| `padding.md` | Expand `paddingSizes_` with op-specific variants (avgpool2d, depthwise conv2d) and emission rules |
| `padding.md` | Document missing `unneededPad*` fields |
| `labeleddatastructure.md` | Remove stale `SDSC-json-api.md` xref from `wordLength` naming note |
| `scheduletreenode.md` | Fix `padding_` field description |
| `sdscbundle-schema.json` | Schema sync with doc fixes |

---

### Task — Link and reference fixes (2026-09-09 / 2026-09-08)

**Commits:** `a195245`, `f8c71ff`, `c16da13`

| File | Change |
|---|---|
| `reference.md` | Fix broken links |
| `Overview.md` | Fix SuperDSC-Bundle spec link |
| `Overview.md` | Remove redundant notes |

---

### Task — MLIR Bundle API corrections (2026-09-07)

**Commits:** `27e8788`, `b0f537b`, `b30de51`, `7860a50`, `0630ff8`, `66f603e`

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Soften `device_mem_allocate` loop placement from MUST to SHOULD |
| `MLIR-bundle-API.md` | Correct `symbol_ids` scoping description |
| `MLIR-bundle-API.md` | Remove misplaced `scf.for` loop bound note from Symbolic Values section |
| `MLIR-bundle-API.md` | Remove implementation-detail parentheticals from `scf.for` descriptions |
| `MLIR-bundle-API.md` | Correct symbolic loop bounds description |
| `MLIR-bundle-API.md` | Broaden bundle container operands description to include symbolic dimension sizes |

---

### Task — Symbolic values documentation and MLIR example fixes (2026-09-05)

**Commits:** `33b5bad`, `57e023e`, `1e4d678`

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Add symbolic values and addresses documentation; fix `device_mem_allocate` loop constraint contradiction |
| `MLIR-bundle-usage-examples.md` | Add symbolic address and dimension examples; fix example inconsistencies |
| `MLIR-complete-example.md` | Fix MLIR example inconsistencies |

---

### Task — MLIR Bundle API expansion (2026-09-04 / 2026-08-27)

**Commits:** `102848d`, `2f57043`, `512e04b`

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Document bundle parameters; add `input_arg_extract`; expand operations table; reformat to formal API reference style |
| `MLIR-bundle-usage-examples.md` | Add `sdscbundle.device_mem_allocate` examples |

---

### Task — Reference and stage restructuring (2026-09-03 / 2026-09-02)

**Commits:** `36228ac`, `61cba1d`, `36f8bb4`, `e6a8ca8`, `34a52bc`, `ea55e63`, `de23860`, `a1ecdbc`

| File | Change |
|---|---|
| `reference.md` | Add KTIR RFC link; fix links to public repos; add torch-spyre; remove glossary section |
| `README.md` | Remove error handling and best practices stages; restructure into Introduction / MLIR / JSON stages; reorganize into staged learning order |
| `SDSC-json-api.md` | Add JSON object hierarchy diagram |
| Multiple | Sync markdown files with `sdscbundle-schema2.json` |

---

### Task — JSON object hierarchy and navigation (2026-09-02)

**Commits:** `8cacc1d`, `26d7566`, `e73d512`, `a1a114f`, `3d56c9e`

| File | Change |
|---|---|
| `JSON-object-Hierarchy.md` | Update to match `sdscbundle-schema2.json` |
| All | Add Previous navigation links to all file footers |
| `foldmanager.md` | Merge `folding.md` into `foldmanager.md`; absorb schedule tree field details |
| `foldproperty.md` | Move folding concept intro from `foldmanager.md` |

---

### Task — Per-object documentation quality pass (2026-08-14)

**Commits:** `cdd182c`, `5e77086`, `a49f1e8`, `1fde3b6`, `f885cdb`, `bb613ba`, `799b862`, `2c25a49`, `df95ec6`, `ddd5e57`, `fac2362`, `da80514`, `40632b2`, `b75de4a`, `ee7f76f`, `eba3f9b`, `ed4d18f`

| File | Change |
|---|---|
| `primarydsinfo.md`, `scheduletreenode.md` | Improve to documentation standard |
| `datastructdims.md`, `datastageparam.md` | Improve to documentation standard |
| `designspaceconfig.md` | Improve; remove Generated by / Written by annotations |
| `foldmanager.md` | Improve; update schema FoldManager descriptions |
| `foldproperty.md` | Improve; update schema FoldProperty descriptions |
| `superdsc-object.md` | Improve; reorder TOC; update nav chain; absorb `root-structure.md` and `wrappeddesignspaceconfig.md` |
| `constantinfo.md` | Improve; update schema ConstantInfo |
| `computeoperation.md` | Merge `operationattributes.md`; apply compliance findings F1–F13 |
| `memoryorganization.md` | Apply compliance findings F1–F8 |
| `labeleddatastructure.md` | Add intro and Context section; apply compliance findings F3–F13 |
| `coordinateinfo.md` | Improve with context, fields table, semantic descriptions, and example annotation |
| `coordinatecontainer.md` | Improve with context, fields table, and example |

---

### Task — Initial directory creation (2026-08-13)

**Commit:** `3317e79`

All API documentation files created for the first time under `sdsc_BUNDLE_APIs/`.
