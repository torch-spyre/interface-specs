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

### Task — Document nonUnifiedAllocInHBM_, coreIdToDscSchedule defaults, remove backend padding_, and clarify fold levels (2026-09-17)

| File | Change |
|---|---|
| `sdscbundle-schema.json` | Add `nonUnifiedAllocInHBM_` boolean/integer(0/1) definition to `ScheduleTreeNode`. Remove backend-only `padding_` property. |
| `scheduletreenode.md` | Add `nonUnifiedAllocInHBM_` to Structure and Fields tables with dedicated explanation of unified vs. non-unified HBM layout. Remove backend-only `padding_` field and duplicate inline `CoordinateInfo` and `folds hierarchy` tables. |
| `JSON-object-Hierarchy.md` | Add `nonUnifiedAllocInHBM_` to `ScheduleTreeNode` branch. Remove `padding_`. Document 4-element `coreIdToDscSchedule` step tuple breakdown. |
| `json-schema.md` | Update schema version history to record the addition of `nonUnifiedAllocInHBM_`. |
| `superdsc-object.md` | Add Schedule step tuple reference table defining `[datadsc_idx, dldsc_idx, before_sync, after_sync]` with `datadsc_idx = -1` default, `dldsc_idx` index into `dscs_`, and barrier flags. |
| `SDSC-json-api.md` | Update step 1 to clarify `coreFoldProp_.factor_` is set to a value between 1 and the maximum number of cores in use (resolving B-1). Update step 2 to describe schedule tuple breakdown and default `[-1, 0, 0, 0]`. |
| `coordinateinfo.md` | Clarify semantic definitions of `spatial`, `temporal`, and `elemArr` levels without confusing "typically 3" assertions. Cross-reference `foldproperty.md#folds-hierarchy`. |
| `foldproperty.md` | Add dedicated `## Folds Hierarchy` section documenting `core_fold`, `corelet_fold`, `row_fold`, `elem_arr_0`, and `elem_arr_1`. |
| `complete-example.md`, `indirect-access-example.md` | Update example `coreIdToDscSchedule` tuples from `[[0, 0, 0, 0]]` to standard default `[[-1, 0, 0, 0]]`. |
| `datastructdims.md`, `sdscbundle-schema.json` | Remove deprecated dimension fields (`r_`, `c_`, `ij_`, `rc_`, `kij_`, `sij_`, `zij_`, `si_`, `sj_`, `zi_`, `zj_`). |
| `padding.md`, `coordinateinfo.md` | Remove duplicate CoordinateInfo padding section from `padding.md` and consolidate enum value descriptions directly into `coordinateinfo.md`. |
| `figures/gen_hierarchy.py`, `figures/sdsc_json_hierarchy.png` | Add `nonUnifiedAllocInHBM_` to `ScheduleTreeNode` diagram node and re-render diagram. |

---

### Task — Clarify symbol ID uniqueness rule for scf.for loop bodies (M-7) (2026-09-16)

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Add loop-reuse exception to symbol ID uniqueness rule in three locations: `symbol_ids` attribute description, Constraints block, and Semantic constraints section. Clarifies that the same symbol IDs may be reused across `scf.for` iterations because the loop iterator provides per-iteration uniqueness. M-7 closed as false positive. |

---

### Task — Fix tensor naming convention in complete-example.md (B-3) (2026-09-16)

| File | Change |
|---|---|
| `complete-example.md` | Rename tensors from bare `"input"` / `"output"` to `"gelu-Tensor0"` / `"gelu-Tensor1"` to match the `"<op>-Tensor<N>"` convention used in all other examples. Update `inputLabeledDs` and `outputLabeledDs` references in `computeOp_` accordingly (`"gelu-Tensor0-idx0"`, `"gelu-Tensor1-idx1"`). |

---

### Task — Align softmax decomposition to 6-step max/sub/exp/sum/reciprocal/mul (M-11) (2026-09-16)

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Replace 5-step softmax example (`maxnonstick`, `sumnonstick`, `realdiv`) in the `sdscbundle.sdsc_execute` sequential operations example with the canonical 6-step decomposition (`max`, `sub`, `exp`, `sum`, `reciprocal`, `mul`) used in `MLIR-bundle-usage-examples.md` and `MLIR-complete-example.md` |

---

### Task — Add dataStageParam_ and coordinates_ to complete-example.md (M-6) (2026-09-16)

| File | Change |
|---|---|
| `complete-example.md` | Add `dataStageParam_` (key `"0"`, name `"core"`, `ss_`/`el_` = `{mb_: 16, out_: 128}` — even 2-way split). Add `coordinates_` to both allocate nodes: `mb` as 2-way core-split non-stick dimension (`core_fold factor=2, alpha=16, elem_arr_0 factor=16`); `out` as unsplit stick dimension (`core_fold factor=1, alpha=128, elem_arr_0 factor=64, elem_arr_1 factor=2`). Add missing `prev_: "input_alloc"` on `output_alloc` node. |

---

### Task — Restore spec-accurate broadcast stick constraint wording (M-3) (2026-09-16)

| File | Change |
|---|---|
| `stick-layout-constraints.md` | Restore Unary and Broadcast Operations constraints to match `SuperDSC-Bundle.md` verbatim: split the collapsed single bullet back into three sentences — (1) any layout acceptable but all tensors must share it, (2) if any stick dimension has broadcast in a tensor all stick dimensions of that tensor must have broadcast, (3) if a stick dimension has broadcast in all tensors its size must equal the number of elements one full stick would have. Remove the erroneous "with only one valid element" qualifier that was not in the spec. |

---

### Task — Replace remaining DL16 format references with SEN169_FP16 in stick-layout-constraints.md (2026-09-16)

| File | Change |
|---|---|
| `stick-layout-constraints.md` | Replace 6 remaining `DL16` format references with `SEN169_FP16`: BatchMatmul output format, Input1/Input2 per-precision stick entries, quantization input stick constraint, and up-casting operation descriptions. OpFunc identifier strings (`DL16TOFP32`, `FP32TODL16`, `FP8TODL16`) in `computeoperation.md` are left unchanged as they are normative JSON string values. |

---

### Task — Remove ReStickifyOpLx reference from stick-layout-constraints.md (2026-09-16)

| File | Change |
|---|---|
| `stick-layout-constraints.md` | Remove parenthetical sentence mentioning `ReStickifyOpLx` from the Stick Altering Data Shuffle section |

---

### Task — Replace all DF16 references with SEN169_FP16 in API docs (2026-09-16)

| File | Change |
|---|---|
| `computeoperation.md` | Replace all 90 occurrences of `DF16` with `SEN169_FP16` in the Supported Operations table (Precision and Notes columns) |
| `stick-layout-constraints.md` | Replace 1 occurrence of `DF16` with `SEN169_FP16` in the Stick Altering Data Shuffle constraints |

---

### Task — Add json-schema.md wrapper page for sdscbundle-schema.json (2026-09-16)

| File | Change |
|---|---|
| `json-schema.md` | **New file.** Markdown wrapper for `sdscbundle-schema.json`: describes the schema's role and enforcement scope, reproduces the version history from the schema's `$comment` field, links to the raw JSON file, and carries the standard navigation footer |
| `README.md` | Update section 3.3 to link to `json-schema.md` instead of the raw `sdscbundle-schema.json` |
| `JSON-object-Hierarchy.md` | Update "Next" footer to point to `json-schema.md` |
| `superdsc-object.md` | Update "Previous" footer to point to `json-schema.md` |
| `Overview.md` | Update schema link to point to `json-schema.md` |
| `SDSC-json-api.md` | Update schema link to point to `json-schema.md` |

---

### Task — Add sdscbundle-schema.json to README Table of Contents (2026-09-16)

| File | Change |
|---|---|
| `README.md` | Add section 3.3 "JSON Schema (normative)" linking to `sdscbundle-schema.json` in Stage 3 — JSON Layer Fundamentals |

---

### Task — Overview.md typo fixes, remove Current Limitation note, trim README header (2026-09-16)

| File | Change |
|---|---|
| `Overview.md` | Fix typos: "stand" → "stands", "JSON based" → "JSON-based", "a sets" → "a set", "user written" → "user-written", "bundle.mir" → "bundle.mlir", "json file" → "JSON file", sentence-case "Figure" → "figure" and "A full list" → "a full list", "tiling structure define" → "tiling structure defines" |
| `Overview.md` | Remove "Current Limitation" bullet from the Tensor Allocation vs Work Division section — the backend now supports cross-core data movement |
| `README.md` | Remove `Last Updated` and `Git Baseline` fields from the header; retain only `Spec Version` and `Status` |

---

### Task — Add schema reference, indirect-access example, and navigation fixes (2026-09-16)

| File | Change |
|---|---|
| `Overview.md` | Add paragraph under "API Components" stating that all `sdsc_*.json` files must conform to `sdscbundle-schema.json`; link to the schema file; distinguish structural (schema) from semantic (doc) constraints |
| `SDSC-json-api.md` | Add blockquote callout at the top of the Introduction linking to `sdscbundle-schema.json` as the normative structural reference |
| `indirect-access-example.md` | **New file.** End-to-end JSON example for a single-core `topkvalue` (Top-K gather) operation demonstrating indirect-access tensors: four-tensor pattern (`INPUT`, `OUTPUT`, `KERNEL` value tensor, `KERNEL_IDX` index tensor); `indirectAllocType_` on both allocate nodes; `relatedIndirectAccessAlloc_` cross-links; `indexTensorType_: "index"`; `maxDimSizes_` page size on value tensor; `indirectAccessIndexLabeledDs` on `ComputeOperation`; fully filled `coordinates_` and `startAddressCoreCorelet_` on all nodes; annotation section and memory-layout table |
| `README.md` | Add section 7.2 "Indirect Access Example (JSON)" pointing to `indirect-access-example.md` in Stage 7 — Worked Examples |
| `complete-example.md` | Update navigation footer "Next" to point to `indirect-access-example.md` |
| `reference.md` | Update navigation footer "Previous" to point to `indirect-access-example.md` |

---

### Task — Add Core Work Division Constraints section to stick-layout-constraints.md (2026-09-15)

| File | Change |
|---|---|
| `stick-layout-constraints.md` | Add `## Core Work Division Constraints` section with three subsections: **Data Tensors** (stick-multiple alignment, 256 MB DDR span, dtype note), **Index Tensors** (integral-sticks or sub-stick rule), **Reduction Operations with Multiple Reduction Dimensions** (only one reduction dim may be split). Sourced from `SuperDSC-Bundle.md` lines 399–405 and `work_division.txt` / `work_division_analysis.md` in `.ignore/`. `spyre_hint` frontend API excluded as it is transparent to the SDSC artifacts. |

Commit: `62b75c8`

---

### Task — Add Validation sections to MLIR-bundle-API.md and SDSC-json-api.md (2026-09-09)

| File | Change |
|---|---|
| `MLIR-bundle-API.md` | Add `## Validation` section at end of reference content: structural constraints (permitted dialects, operand/symbol_ids count, device_mem_allocate size, input_arg_extract source type, func.func parameter types) checked by the MLIR verifier; semantic constraints (constant loop bounds, unique symbol IDs, resolvable sdsc_filename paths) checked during pipeline processing |
| `SDSC-json-api.md` | Add `## Validation` section at end of reference content: schema validation via `sdscbundle-schema.json`; semantic validation (numCoresUsed_ / coreIdsUsed_ consistency, coreIdToDsc_ index validity, symbol ID / operand correspondence, stick-size work multiple, 256 MB DDR span limit) |

---

### Task — Add Constraints column to coordinatecontainer.md fields table (2026-09-09)

| File | Change |
|---|---|
| `coordinatecontainer.md` | Add missing Constraints column to Fields table to match the standard used by all other object docs; move key-pattern constraint from Description into Constraints for `coordInfo`; add outer-key constraint for `coreIdToWkSlice_`. Resolves gap analysis item 12. |

---

### Decision — Item 11 closed as false positive (2026-09-09)

Gap analysis item 11 (orphaned/misaligned heading structure in `MLIR-bundle-API.md`) investigated and closed as a false positive. The three `#` characters flagged as headings are MLIR affine map alias definitions (e.g. `#map_name = affine_map<...>`) inside fenced code blocks — not Markdown headings. The document's actual heading hierarchy (`#` → `##` → `###`) is correct and well-formed. No changes required.

---

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
