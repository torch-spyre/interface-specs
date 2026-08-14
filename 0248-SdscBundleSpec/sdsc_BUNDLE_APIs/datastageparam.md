# DataStageParam

`DataStageParam` specifies the tensor tile dimensions for a single core at
two execution stages: **steady-state** (`ss_`) and **epilogue** (`el_`).
When work is divided evenly across all time steps, `ss_` and `el_` are
identical. When the final time step processes a smaller tile (e.g. a
minibatch of 5 split across 2 cores gives 3 and 2), `el_` captures that
smaller size for the epilogue step.

## Context

`DataStageParam` entries appear inside `DesignSpaceConfig.dataStageParam_`,
keyed by core ID (as a string integer). There is typically one entry with key
`"0"` covering all cores, but each core can have its own entry when work
differs per core.

```json
"dataStageParam_": {
  "0": { <DataStageParam> }
}
```

Each value (`ss_` and `el_`) is a [`DataStructDims`](datastructdims.md)
object containing the tile size for that stage. For padding details in
windowed operations such as convolution, `ss_` and `el_` include a
`paddingSizes_` sub-field. See [Padding](padding.md).

## Structure

```json
{
  "name_": "<string>",
  "ss_":   <DataStructDims>,
  "el_":   <DataStructDims>
}
```

## Fields

`ss_` and `el_` are required. No additional properties are allowed.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `name_` | string | No | `"core"` `"corelet"` `"row"` | Stage name identifying the hierarchy level this param applies to. |
| `ss_` | [DataStructDims](datastructdims.md) | Yes | — | Steady-state tile dimensions — the tile size used by most time steps. |
| `el_` | [DataStructDims](datastructdims.md) | Yes | — | Epilogue tile dimensions — the tile size for the final time step when work does not divide evenly. Equal to `ss_` when work is uniform. |

## Example

### Uniform work division (ss_ == el_)

A 2-core matmul where both cores process equal tiles in every time step:

```json
"dataStageParam_": {
  "0": {
    "name_": "core",
    "ss_": {"mb_": 16, "out_": 64, "in_": 128},
    "el_": {"mb_": 16, "out_": 64, "in_": 128}
  }
}
```

### Non-uniform work division (ss_ ≠ el_)

A minibatch of 5 split across 2 time steps (3 + 2). Steady-state processes
3 elements; the epilogue processes 2:

```json
"dataStageParam_": {
  "0": {
    "name_": "core",
    "ss_": {"mb_": 3, "out_": 64, "in_": 128},
    "el_": {"mb_": 2, "out_": 64, "in_": 128}
  }
}
```

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: PrimaryDsInfo →](primarydsinfo.md) |
|:--|:--:|--:|
