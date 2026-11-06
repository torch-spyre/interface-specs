# MemoryOrganization

`MemoryOrganization` specifies which physical memories hold a tensor on the
Spyre device. It appears as the `memOrg_` field inside every
[`LabeledDataStructure`](labeleddatastructure.md) and controls whether the
tensor is allocated in High Bandwidth Memory (HBM), the per-core LX
scratchpad, or both.

## Context

`memOrg_` is a required field of every [`LabeledDataStructure`](labeleddatastructure.md)
entry in `labeledDs_`. The corresponding [`ScheduleTreeNode`](scheduletreenode.md)
allocate node's `component_` field (`"hbm"` or `"lx"`) MUST be consistent
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
    "isPresent":    <bool | 0 | 1>,
    "isPadded":     <bool | 0 | 1>,
    "isZeroPadded": 0 | 1
  },
  "lx": {
    "isPresent":    <bool | 0 | 1>,
    "isPadded":     <bool | 0 | 1>,
    "isZeroPadded": 0 | 1
  }
}
```

## Fields

Both `hbm` and `lx` are optional keys; omitting a key is equivalent to
`"isPresent": 0`. No additional properties are allowed inside either sub-object.

`isPresent` and `isPadded` are flags: each accepts a JSON boolean or the
integers `1`/`0`, with the same meaning. `isZeroPadded` is not a flag and
accepts only the integers `0` and `1`.

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `hbm` | object | No | — | High Bandwidth Memory slot. Omit or set `isPresent: 0` when the tensor is not in HBM. |
| `hbm.isPresent` | boolean or integer (0/1) | Yes (if `hbm` present) | `true`/`false` or `1`/`0` | `true` (`1`) = tensor resides in HBM. MUST match `component_: "hbm"` in the corresponding `ScheduleTreeNode`. |
| `hbm.isPadded` | boolean or integer (0/1) | No | `true`/`false` or `1`/`0` | `true` (`1`) = HBM allocation includes padding. See [Padding](padding.md). |
| `hbm.isZeroPadded` | integer | No | 0 or 1 | 1 = padded region in HBM is zero-filled. Only meaningful when `isPadded` is `true` (`1`). |
| `lx` | object | No | — | LX per-core local scratchpad slot. Omit or set `isPresent: 0` when the tensor is not in LX. |
| `lx.isPresent` | boolean or integer (0/1) | Yes (if `lx` present) | `true`/`false` or `1`/`0` | `true` (`1`) = tensor resides in the LX scratchpad. MUST match `component_: "lx"` in the corresponding `ScheduleTreeNode`. |
| `lx.isPadded` | boolean or integer (0/1) | No | `true`/`false` or `1`/`0` | `true` (`1`) = LX allocation includes padding. See [Padding](padding.md). |
| `lx.isZeroPadded` | integer | No | 0 or 1 | 1 = padded region in LX is zero-filled. Only meaningful when `isPadded` is `true` (`1`). |

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

[↑ Spec Map](../../SuperDSC-Bundle.md#spec-map)|
