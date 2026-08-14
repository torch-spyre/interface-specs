# LabeledDataStructure

A `LabeledDataStructure` describes a single physical tensor used by a
[`DesignSpaceConfig`](designspaceconfig.md). It carries the tensor's role
(`INPUT`, `OUTPUT`, `KERNEL`, or `KERNEL_IDX`), data format, per-dimension
scale factors, word length, and memory residency. Both input and output tensors
must be listed in `labeledDs_`.

## Context

`LabeledDataStructure` entries appear in the `labeledDs_` array inside each
[`DesignSpaceConfig`](designspaceconfig.md). The `ldsIdx_` field is the
zero-based index into that array; [`ScheduleTreeNode`](scheduletreenode.md)
allocate nodes reference tensors by the same index. Compute operations
reference individual tensors by the composite name
`"<dsName_>-idx<ldsIdx_>"` (e.g. `"convolution-Tensor0-idx0"`).

```json
"labeledDs_": [
  { <LabeledDataStructure> },
  ...
]
```

## Structure

```json
{
  "ldsIdx_":     <int>,
  "dsName_":     "<string>",
  "dsType_":     "INPUT" | "OUTPUT" | "KERNEL" | "KERNEL_IDX",
  "scale_":      [<number>, ...],
  "wordLength":  <int>,
  "dataFormat_": "<format>",
  "memOrg_":     <MemoryOrganization>
}
```

## Fields

All five required fields must be present. No additional properties are allowed.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `ldsIdx_` | integer | Yes | >= 0 | Zero-based index into the parent `labeledDs_` array. Referenced by `ScheduleTreeNode.ldsIdx_` and used to form the compute-op tensor name suffix `-idx<N>`. |
| `dsName_` | string | Yes | — | Unique tensor name within the DSC. Compute operations reference this tensor as `"<dsName_>-idx<ldsIdx_>"` (e.g. `"convolution-Tensor0-idx0"`). |
| `dsType_` | string enum | Yes | `"INPUT"` `"OUTPUT"` `"KERNEL"` `"KERNEL_IDX"` | Tensor role. Must match a type defined in `primaryDsInfo_` of the parent DSC. Sourced from torch_spyre `LAYOUT_INPUT_LABELS` / `LAYOUT_OUTPUT_LABELS`. |
| `scale_` | array of number | No | One entry per dimension in `dsType_` order | Per-dimension scale factors in `layoutDimOrder_` order. `1` = normal size; `-1` = reduced / broadcast; `-2` = reduced / broadcast stick dimension. |
| `wordLength` | integer | No | >= 1 | Element size in bytes (e.g. `2` for FP16). **Note:** field name lacks the trailing underscore used by all other fields — see [F8 note](#f8-note-on-wordlength-naming). |
| `dataFormat_` | string enum | Yes | see [Data Formats](#data-formats) | Numeric format of each tensor element. `SEN169_FP16` is the most common Spyre format. |
| `memOrg_` | [MemoryOrganization](memoryorganization.md) | Yes | — | Memory residency — specifies which memories (HBM, LX scratchpad) hold this tensor. |

## Data Formats

The schema enforces the following Spyre-validated subset. The full list of formats
supported by the torch_spyre runtime (including `IEEE_FP32`, `BFLOAT16`,
`SEN121_FP4`, `BOOL`, and others) is documented in
[SDSC-json-api.md](SDSC-json-api.md).

| Value | Description |
|---|---|
| `SEN169_FP16` | SEN169 16-bit floating point (most common for Spyre) |
| `SEN169_BFP16` | SEN169 brain float 16 |
| `SEN169_FP32` | SEN169 32-bit floating point |
| `SEN169_INT8` | SEN169 8-bit integer |
| `SEN169_INT16` | SEN169 16-bit integer |
| `SEN169_INT32` | SEN169 32-bit integer |

## F8 Note on `wordLength` Naming

All other fields in `LabeledDataStructure` use a trailing underscore
(`ldsIdx_`, `dsName_`, etc.). The field `wordLength` does not.
`SDSC-json-api.md` refers to it as `wordLength_`. The serialized key in the
JSON bundle is `wordLength` (no underscore) — this inconsistency is a known
anomaly. Do not add a trailing underscore when writing bundle JSON.

## Example

The example below shows an INPUT tensor placed in `labeledDs_` and how the
same tensor is referenced by name in `computeOp_`.

```json
"labeledDs_": [
  {
    "ldsIdx_": 0,
    "dsName_": "convolution-Tensor0",
    "dsType_": "INPUT",
    "scale_": [1.0, 1.0, 1.0],
    "wordLength": 2,
    "dataFormat_": "SEN169_FP16",
    "memOrg_": {
      "hbm": {"isPresent": 1},
      "lx":  {"isPresent": 0}
    }
  }
],
"computeOp_": [
  {
    "exUnit": "sfp",
    "opFuncName": "convolution",
    "inputLabeledDs":  ["convolution-Tensor0-idx0"],
    "outputLabeledDs": ["convolution-Tensor1-idx1"]
  }
]
```

`"convolution-Tensor0-idx0"` is formed by concatenating `dsName_`
(`"convolution-Tensor0"`) with `-idx` and `ldsIdx_` (`0`).

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: MemoryOrganization →](memoryorganization.md) |
|:--|:--:|--:|
