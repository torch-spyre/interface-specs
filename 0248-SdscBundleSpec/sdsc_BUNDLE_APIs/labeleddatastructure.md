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
  "wordLength":  <number>,
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
| `wordLength` | number | No | > 0 (exclusive) | Element size in bytes. Integer for full-byte formats (e.g. `2` for FP16); fractional for sub-byte formats (e.g. `0.5` for 4-bit types such as `SENINT4` and `SEN121_FP4`). **Note:** field name lacks the trailing underscore used by all other fields — see [F8 note](#f8-note-on-wordlength-naming). |
| `dataFormat_` | string enum | Yes | see [Data Formats](#data-formats) | Numeric format of each tensor element. `SEN169_FP16` is the most common Spyre format. |
| `memOrg_` | [MemoryOrganization](memoryorganization.md) | Yes | — | Memory residency — specifies which memories (HBM, LX scratchpad) hold this tensor. |

## Data Formats

| Value | Description |
|---|---|
| `SEN169_FP16` | SEN169 16-bit floating point (most common for Spyre) |
| `IEEE_FP32` | IEEE 754 32-bit floating point |
| `IEEE_FP16` | IEEE 754 16-bit floating point |
| `BFLOAT16` | Brain float 16 |
| `BOOL` | Boolean |
| `SEN143_FP8` | SEN143 8-bit floating point |
| `SEN152_FP8` | SEN152 8-bit floating point |
| `SEN153_FP9` | SEN153 9-bit floating point |
| `SEN18F_FP24` | SEN18F 24-bit floating point |
| `SEN080_FP8` | SEN080 8-bit floating point |
| `SEN053_FP8` | SEN053 8-bit floating point |
| `SEN121_FP4` | SEN121 4-bit floating point |
| `SENINT2` | SEN 2-bit integer |
| `SENINT4` | SEN 4-bit integer |
| `SENINT8` | SEN 8-bit integer |
| `SENINT16` | SEN 16-bit integer |
| `SENINT24` | SEN 24-bit integer |
| `SENUINT2` | SEN 2-bit unsigned integer |
| `SENUINT32` | SEN 32-bit unsigned integer |
| `IEEE_INT32` | IEEE 32-bit signed integer |
| `IEEE_INT64` | IEEE 64-bit signed integer |

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

| [← Previous: DesignSpaceConfig](designspaceconfig.md) | [↑ Table of Contents](README.md) | [Next: MemoryOrganization →](memoryorganization.md) |
|:--|:--:|--:|
