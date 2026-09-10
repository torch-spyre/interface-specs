# DataStructDims

`DataStructDims` carries the dimension sizes for a tensor at a given level of
the compute hierarchy (total, per-core, per-corelet, etc.). It appears as the
`N_` field in [`DesignSpaceConfig`](designspaceconfig.md) (total un-tiled
dimensions) and as `ss_` / `el_` inside
[`DataStageParam`](datastageparam.md) (steady-state and epilogue tile sizes).
All numeric dimension values are doubles; `-1` means the dimension is unset
for this object.

## Context

`DataStructDims` appears in three places:

| Parent field | Meaning |
|---|---|
| `DesignSpaceConfig.N_` | Total (un-tiled) tensor dimensions for the operation |
| `DataStageParam.ss_` | Per-core steady-state tile dimensions |
| `DataStageParam.el_` | Per-core epilogue tile dimensions (differs from `ss_` only when work is not evenly divided) |

## Structure

```json
{
  "name_":  "<string>",
  "in_":    <number>,
  "out_":   <number>,
  "mb_":    <number>,
  "i_":     <number>,
  "j_":     <number>,
  "ki_":    <number>,
  "kj_":    <number>,
  "x_":     <number>,
  "x1_":    <number>,
  "y_":     <number>,
  "symbolicDimInfo_":  <map<string, SymbolicDimInfo>>,
  "maxSymbolicVolume_": <map<string, int>>,
  "coreletSplit_":     <map<string, array<int>>>,
  "rowSplit_":         <map<string, map<string, array<int>>>>,
  "peSfpSplit_":       <map<string, map<string, map<string, int>>>>,
  "paddingSizes_":     <map<string, DimPaddingSizes>>
}
```

## Fields

No fields are required; include only the dimensions relevant to the operation.

**Note on dim field values:** All numeric dimension fields accept either a number (typically `-1` when unset, or a positive value when set) or an empty object `{}`. The empty-object form is emitted in some symbolic-dimension contexts and must be tolerated by readers.

### Primary dimension fields

These are always serialized (value `-1` when unset):

| Field | Type | Description |
|---|---|---|
| `name_` | string | Dimension set name (e.g. `"core"`, `"corelet"`). |
| `in_` | number | Input features / channels. `-1` = unset. |
| `out_` | number | Output features / channels. `-1` = unset. |
| `mb_` | number | Minibatch size. `-1` = unset. |
| `i_` | number | Output image rows (derived from input + padding). `-1` = unset. |
| `j_` | number | Output image cols (derived from input + padding). `-1` = unset. |
| `ki_` | number | Kernel rows. `-1` = unset. |
| `kj_` | number | Kernel cols. `-1` = unset. |
| `x_` | number | Repeat dimension that does not add reuse (e.g. attention heads). `-1` = unset. |
| `x1_` | number | Additional repeat dimension (e.g. attention heads). `-1` = unset. |
| `y_` | number | Kernel reuse dimension (e.g. time step). `-1` = unset. |

### Advanced fields

| Field | Type | Description |
|---|---|---|
| `paddingSizes_` | map&lt;string, DimPaddingSizes&gt; | Per-dimension padding information. Keys are dim names (e.g. `"r_"`, `"c_"`). See [Padding](padding.md). |
| `symbolicDimInfo_` | map&lt;string, SymbolicDimInfo&gt; | Per-dimension symbolic size constraints. Keys are dimension names (e.g. `"mb_"`). See [SymbolicDimInfo](#symbolicdiminfo) below. |
| `maxSymbolicVolume_` | map&lt;string, integer&gt; | Maximum combined symbolic volume for a set of dims. Keys are dim-set arrays encoded as strings (e.g. `'["in_","mb_"]'`). |
| `coreletSplit_` | map&lt;string, array&lt;int&gt;&gt; | Explicit per-dimension work split across corelets. One integer per corelet. |
| `rowSplit_` | map&lt;string, map&lt;string, array&lt;int&gt;&gt;&gt; | Per-dimension, per-corelet work split across PT rows. |
| `peSfpSplit_` | map&lt;string, map&lt;string, map&lt;string, int&gt;&gt;&gt; | Per-dimension, per-corelet work split between PE and SFP components. |

---

## SymbolicDimInfo

`SymbolicDimInfo` constrains a symbolic dimension size at runtime. It appears as
a value inside `DataStructDims.symbolicDimInfo_`, keyed by dimension name
(e.g. `"mb_"`). Both fields are required.

### Structure

```json
{
  "maxSize_":    <integer>,
  "granularity_": <integer>
}
```

### Fields

| JSON field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `maxSize_` | integer | Yes | >= 1 | Upper bound the backend must plan memory for. The runtime value supplied at job launch must not exceed this. Corresponds to the `max_value` annotation on `!sdscbundle.input_arg<index, max_value=N>` in the MLIR bundle. |
| `granularity_` | integer | Yes | >= 1 | Step size the runtime value must be a multiple of. For a dimension split across N cores, `granularity_` must be a multiple of N. Corresponds to the `granularity` annotation on `!sdscbundle.input_arg<index, granularity=N>` in the MLIR bundle. |

### JSON ↔ MLIR name mapping

| JSON (`symbolicDimInfo_`) | MLIR (`!sdscbundle.input_arg` annotation) | MLIR extraction keyword |
|---|---|---|
| `maxSize_` | `max_value=N` | `sdscbundle.input_arg_extract max_value` |
| `granularity_` | `granularity=N` | `sdscbundle.input_arg_extract granularity` |

See [`sdscbundle.input_arg_extract`](MLIR-bundle-API.md#sdscbundleinput_arg_extract) for the full MLIR usage.

### Example

A symbolic `mb_` dimension with a maximum of 32 and a granularity of 1
(runtime value must be a positive integer ≤ 32):

```json
"symbolicDimInfo_": {
  "mb_": {
    "maxSize_":    32,
    "granularity_": 1
  }
}
```

The corresponding MLIR parameter and extractions:

```mlir
func.func @op(%mb_arg: !sdscbundle.input_arg<index, granularity=1, max_value=32>) {
  %mb   = sdscbundle.input_arg_extract value       from %mb_arg
            : !sdscbundle.input_arg<index, granularity=1, max_value=32> -> index
  %gran = sdscbundle.input_arg_extract granularity from %mb_arg
            : !sdscbundle.input_arg<index, granularity=1> -> index
  %max  = sdscbundle.input_arg_extract max_value   from %mb_arg
            : !sdscbundle.input_arg<index, max_value=32> -> index
  ...
}
```

### Deprecated dimension fields

Serialized unless `skipDeprecatedFields=true`. Prefer the primary dims above.

| Field | Description |
|---|---|
| `r_` | Input image rows with zero padding. |
| `c_` | Input image cols with zero padding. |
| `ij_` | Output image rows/cols combined. |
| `rc_` | Input image rows/cols with zero padding combined. |
| `kij_` | Kernel rows/cols combined. |
| `sij_` | Stride rows/cols combined. |
| `zij_` | Zero-pad rows/cols combined. |
| `si_` | Stride along rows. |
| `sj_` | Stride along cols. |
| `zi_` | Zero-pad rows. |
| `zj_` | Zero-pad cols. |

## Example

### DesignSpaceConfig.N_ — total operation dimensions

```json
"N_": {
  "mb_": 32,
  "out_": 128,
  "in_": 64
}
```

### DataStageParam — steady-state vs epilogue tile sizes

When the minibatch is split unevenly across 3 cores (2+2+1), `ss_` holds
the full-size tile (2) and `el_` holds the smaller final tile (1):

```json
"dataStageParam_": {
  "0": {
    "ss_": {"mb_": 2, "out_": 128, "in_": 64},
    "el_": {"mb_": 1, "out_": 128, "in_": 64}
  }
}
```

---

| [← Previous: PrimaryDsInfo](primarydsinfo.md) | [↑ Table of Contents](README.md) | [Next: DataStageParam →](datastageparam.md) |
|:--|:--:|--:|
