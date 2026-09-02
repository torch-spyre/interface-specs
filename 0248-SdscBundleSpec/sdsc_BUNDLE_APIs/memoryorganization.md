# MemoryOrganization

`MemoryOrganization` specifies which physical memories hold a tensor on the
Spyre device. It appears as the `memOrg_` field inside every
[`LabeledDataStructure`](labeleddatastructure.md) and controls whether the
tensor is allocated in High Bandwidth Memory (HBM), the per-core LX
scratchpad, or both.

## Context

`memOrg_` is a required field of every [`LabeledDataStructure`](labeleddatastructure.md)
entry in `labeledDs_`. The corresponding [`ScheduleTreeNode`](scheduletreenode.md)
allocate node's `component_` field (`"hbm"` or `"lx"`) must be consistent
with whichever memory key has `"isPresent": 1` here.

```json
"memOrg_": {
  "hbm": { "isPresent": 1 },
  "lx":  { "isPresent": 0 }
}
```

## Structure

```json
{
  "hbm": {
    "isPresent":    0 | 1,
    "isPadded":     0 | 1,
    "isZeroPadded": 0 | 1
  },
  "lx": {
    "isPresent":    0 | 1,
    "isPadded":     0 | 1,
    "isZeroPadded": 0 | 1
  }
}
```

## Fields

Both `hbm` and `lx` are optional keys; omitting a key is equivalent to
`"isPresent": 0`. No additional properties are allowed inside either sub-object.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `hbm` | object | No | — | High Bandwidth Memory slot. Omit or set `isPresent: 0` when the tensor is not in HBM. |
| `hbm.isPresent` | integer | Yes (if `hbm` present) | 0 or 1 | 1 = tensor resides in HBM. Must match `component_: "hbm"` in the corresponding `ScheduleTreeNode`. |
| `hbm.isPadded` | integer | No | 0 or 1 | 1 = HBM allocation includes padding. See [Padding](padding.md). |
| `hbm.isZeroPadded` | integer | No | 0 or 1 | 1 = padded region in HBM is zero-filled. Only meaningful when `isPadded: 1`. |
| `lx` | object | No | — | LX per-core local scratchpad slot. Omit or set `isPresent: 0` when the tensor is not in LX. |
| `lx.isPresent` | integer | Yes (if `lx` present) | 0 or 1 | 1 = tensor resides in the LX scratchpad. Must match `component_: "lx"` in the corresponding `ScheduleTreeNode`. |
| `lx.isPadded` | integer | No | 0 or 1 | 1 = LX allocation includes padding. See [Padding](padding.md). |
| `lx.isZeroPadded` | integer | No | 0 or 1 | 1 = padded region in LX is zero-filled. Only meaningful when `isPadded: 1`. |

## Example

### Typical: tensor in HBM only

```json
"memOrg_": {
  "hbm": {"isPresent": 1},
  "lx":  {"isPresent": 0}
}
```

### LX scratchpad with zero-padding

```json
"memOrg_": {
  "hbm": {"isPresent": 0},
  "lx":  {"isPresent": 1, "isPadded": 1, "isZeroPadded": 1}
}
```

---

| [← Previous: LabeledDataStructure](labeleddatastructure.md) | [↑ Table of Contents](README.md) | [Next: PrimaryDsInfo →](primarydsinfo.md) |
|:--|:--:|--:|
