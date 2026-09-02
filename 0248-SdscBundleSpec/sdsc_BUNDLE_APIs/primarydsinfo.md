# PrimaryDsInfo

`PrimaryDsInfo` records the physical memory layout for each tensor role
([`dsType_`](labeleddatastructure.md)) used by a
[`DesignSpaceConfig`](designspaceconfig.md). It specifies the order in which
dimensions are laid out in memory, which dimensions form a "stick" (the
innermost contiguous unit of storage), and the size of each stick dimension.
Every tensor type that appears in the `labeledDs_` array should have a
corresponding entry here.

## Context

`PrimaryDsInfo` is the value type of the `primaryDsInfo_` map inside each
[`DesignSpaceConfig`](designspaceconfig.md). Keys are uppercase tensor-role
labels such as `INPUT`, `OUTPUT`, `KERNEL`, or `KERNEL_IDX`.

```json
"primaryDsInfo_": {
  "INPUT":  { <PrimaryDsInfo> },
  "OUTPUT": { <PrimaryDsInfo> },
  "KERNEL": { <PrimaryDsInfo> }
}
```

## Structure

```json
{
  "layoutDimOrder_": ["<dim>", ...],
  "stickDimOrder_":  ["<dim>", ...],
  "stickSize_":      [<int>, ...]
}
```

## Fields

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `layoutDimOrder_` | array of string | No | Dimension names matching those declared in `DataStructDims` | Ordered list of dimension names describing how the tensor is laid out in memory, outermost first. |
| `stickDimOrder_` | array of string | No | Must be a suffix subset of `layoutDimOrder_`; length must equal length of `stickSize_` | The innermost contiguous dimensions that form a stick — the atomic unit of storage transfer. |
| `stickSize_` | array of integer | No | Each integer >= 1; parallel to `stickDimOrder_` | Size of each stick dimension in elements. Position `i` is the size of `stickDimOrder_[i]`. |

### Relationship between `stickDimOrder_` and `stickSize_`

`stickDimOrder_` and `stickSize_` are parallel arrays and must have the same
length. Together they define one contiguous storage tile: for example,
`"stickDimOrder_": ["in"]` with `"stickSize_": [64]` means each stick holds
64 elements along the `in` (input-channel) dimension.

## Example

The snippet below is a complete `primaryDsInfo_` block from a convolution DSC.
The `INPUT` and `OUTPUT` tensors share the same layout but differ in their
stick configuration; the `KERNEL` tensor uses a different outer-dimension order.

```json
"primaryDsInfo_": {
  "INPUT": {
    "layoutDimOrder_": ["mb", "out", "in"],
    "stickDimOrder_":  ["in"],
    "stickSize_":      [64]
  },
  "OUTPUT": {
    "layoutDimOrder_": ["mb", "out", "in"],
    "stickDimOrder_":  ["out"],
    "stickSize_":      [64]
  },
  "KERNEL": {
    "layoutDimOrder_": ["out", "in"],
    "stickDimOrder_":  ["in"],
    "stickSize_":      [64]
  }
}
```

Each entry is later cross-referenced by [`LabeledDataStructure.dsType_`](labeleddatastructure.md)
to establish which layout applies to a given tensor in the `labeledDs_` array.

---

| [← Previous: MemoryOrganization](memoryorganization.md) | [↑ Table of Contents](README.md) | [Next: DataStructDims →](datastructdims.md) |
|:--|:--:|--:|
