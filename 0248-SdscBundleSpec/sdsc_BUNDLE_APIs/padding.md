# Padding

For window/padded operations, such as convolution, padding information should be added to both `N_` and `dataStageParam_` in `sdsc.dscs_[0]`, capturing information about front/back padding, stride, and related kernel dimension. If a padded dimension is chunked across cores, front/back padding should be set to `-1` in the core datastage entry.

When a dimension is padded due to window/padded operations like convolution, details of padding need to be specified via the following fields. `paddingSizes_` is only emitted when `sdsc_spec.padding_sizes` is non-empty — currently **avgpool2d** and **depthwise conv2d**. All other operations (pointwise, matmul, reductions, transpose, etc.) leave it absent entirely.

## (i) paddingSizes_

`paddingSizes_` is a dict keyed by spatial dimension label (e.g. `"i"`, `"j"`). It appears in three locations in the SDSC JSON:

| Location | Value used |
|---|---|
| `N_` (top-level op descriptor) | `sdsc_spec.padding_sizes` |
| `dataStageParam_[0].ss_` (per-core schedule) | `padding_sizes_per_core` if set, else `padding_sizes` |
| `dataStageParam_[0].el_` (per-core element loop) | `padding_sizes_per_core` if set, else `padding_sizes` |

A spatial dimension is omitted entirely from `paddingSizes_` if its kernel window dimension (`ki`/`kj`) was squeezed out of the iteration space (kernel size = 1).

If a padded dimension is chunked across cores, `padFront_` and `padBack_` must be set to `-1` in the core datastage entry.

### Fields

| Field | Type | Description |
|---|---|---|
| `padFront_` | integer | Front padding elements. Default `0`. `-1` when the dimension is chunked across cores. |
| `padBack_` | integer | Back padding elements. Default `0`. `-1` when the dimension is chunked across cores. |
| `unneededPad_` | integer | Total unneeded padding elements (sum of front and back unneeded). **Conv2d only** — absent in the avgpool2d variant. |
| `unneededPadFront_` | integer | Unneeded elements absorbed from the front pad. **Conv2d only** — absent in the avgpool2d variant. |
| `unneededPadBack_` | integer | Unneeded elements absorbed from the back pad. **Conv2d only** — absent in the avgpool2d variant. |
| `totalSize_` | integer or `"N/A"` | Total input size inclusive of padding. `"N/A"` when `windowDim_` is unset and `padFront_` < 0. For avgpool2d: `(out - 1) × stride + kernel`; for conv2d: full or per-core input span depending on location. |
| `stride_` | integer ≥ 1 | Stride applied with the operation. Default `1`. |
| `dilation_` | integer ≥ 1 | Dilation applied with the operation. Default `1`. |
| `windowDim_` | string | Associated window primary dimension name (e.g. `"ki_"`). Uses `primaryDimToString` mapping. |

The dimension's size as specified in `N_` and `dataStageParam_` corresponds to the **output** size — e.g. `128` with padding=1 and `126` without padding for a 128×128 input.

### Per-op variants

#### avgpool2d

Each surviving spatial dim (`"i"`, `"j"`) emits the base set of fields. The `unneededPad*` fields are **not** emitted. `padding_sizes_per_core` is empty for avgpool2d, so `generate_sdsc` uses `padding_sizes` for both `ss_` and `el_`.

```json
"paddingSizes_": {
  "i": {
    "padFront_":  <pH>,
    "padBack_":   <pH>,
    "totalSize_": <(out-1)*stride + kernel>,
    "stride_":    <sH>,
    "dilation_":  1,
    "windowDim_": "ki_"
  },
  "j": {
    "padFront_":  <pW>,
    "padBack_":   <pW>,
    "totalSize_": <(out-1)*stride + kernel>,
    "stride_":    <sW>,
    "dilation_":  1,
    "windowDim_": "kj_"
  }
}
```

#### depthwise conv2d

Two separate variants are emitted: a top-level `padding_sizes` (full output size, used for `N_`) and a `padding_sizes_per_core` (per-core output slice, used for `ss_` and `el_`). Each surviving spatial dim includes the three `unneededPad*` fields in addition to the base set.

```json
"paddingSizes_": {
  "i": {
    "padFront_":         <effective front pad after unneeded reduction>,
    "padBack_":          <effective back pad after unneeded reduction>,
    "unneededPad_":      <total_size - min_required_input>,
    "unneededPadFront_": <portion of unneeded pad absorbed from front>,
    "unneededPadBack_":  <portion of unneeded pad absorbed from back>,
    "totalSize_":        <full or per-core input span>,
    "stride_":           <stride_i>,
    "dilation_":         <dilation_i>,
    "windowDim_":        "ki_"
  }
}
```

## (ii) padding field in CoordinateInfo

The `padding` field inside `coordinates_.coordInfo[<dim>]` (a `CoordinateInfo` object within a `scheduleTree_` node) can take on the following values:

```
nopad
lowered_padded
padded_nozeropad
padded_wzeropad
padded_fullspan
padded_fullspan_wunneeded
```

**`padded_nozeropad`** — to be used with conv2d when padding is non-zero.
**`padded_fullspan_wunneeded`** — to be used with conv2d when padding is zero.

---

| [← Previous: FoldManager](foldmanager.md) | [↑ Table of Contents](README.md) | [Next: Stick Layout Constraints →](stick-layout-constraints.md) |
|:--|:--:|--:|
