# Padding

For window/padded operations, such as convolution, padding information must be added to both `N_` and `dataStageParam_` in `sdsc.dscs_[0]`, capturing information about front/back padding, stride, and related kernel dimension. If a padded dimension is chunked across cores, front/back padding must be set to `-1` in the core datastage entry.

When a dimension is padded due to window/padded operations like convolution, details of padding need to be specified via the following fields. `paddingSizes_` is only emitted when `sdsc_spec.padding_sizes` is non-empty — currently convolution and pooling families of operators (**avgpool2d**, **maxpool2d**, **conv2d**, and **depthwise conv2d**). All other operations (pointwise, matmul, reductions, transpose, etc.) leave it absent entirely.

## paddingSizes_

`paddingSizes_` is a dict keyed by spatial dimension label (e.g. `"i"`, `"j"`). It appears in three locations in the SDSC JSON:

| Location | Value used |
|---|---|
| `N_` (top-level op descriptor) | `sdsc_spec.padding_sizes` |
| `dataStageParam_[0].ss_` (per-core schedule) | `padding_sizes_per_core` in the steady-state phase if set, else `padding_sizes` |
| `dataStageParam_[0].el_` (per-core element loop) | `padding_sizes_per_core` in the epilogue phase if set, else `padding_sizes` |

If a padded dimension is chunked across cores, `padFront_` and `padBack_` must be set to `-1` in the core datastage entry (that is, in the entries under `dataStageParam_[0]`).

### Fields

| Field | Type | Description |
|---|---|---|
| `padFront_` | integer | Front padding elements. Default `0`. `-1` (in `dataStageParam_` entries only) when the dimension is chunked across cores. |
| `padBack_` | integer | Back padding elements. Default `0`. `-1` (in `dataStageParam_` entries only) when the dimension is chunked across cores. |
| `unneededPad_` | integer | Total unused elements at the tail of the allocated span. Counts both padding and real data — see the warning below. |
| `unneededPadFront_` | integer | The portion of `unneededPad_` that was taken out of front padding (top/left). |
| `unneededPadBack_` | integer | The portion of `unneededPad_` that was taken out of back padding (bottom/right). |
| `totalSize_` | integer or `"N/A"` | Total input size inclusive of padding. `"N/A"` when `windowDim_` is unset and `padFront_` < 0. It is given by `(out - 1) × stride + kernel + unneededPad_`; it is the full or per-core input span depending on location. |
| `stride_` | integer ≥ 1 | Stride applied with the operation. Default `1`. |
| `dilation_` | integer ≥ 1 | Dilation applied with the operation. Default `1`. |
| `windowDim_` | string | Associated window primary dimension name (e.g. `"ki"`). Uses `primaryDimToString` mapping. |

Some definitions of terms used in computing `unneededPad*` fields:

```
valid_size     = the unpadded input extent along this dimension
                 (the real data, excluding any declared padding)
allocated_span = valid_size + padFront + padBack     # what is laid out in memory
consumed_span  = (out_dim - 1) * stride + window     # what the windows read
unneededPad_   = allocated_span - consumed_span      # the slack
```

where `padFront`/`padBack` are the **originally requested** padding amounts and
`out_dim` is this dimension's size as carried in `N_`/`dataStageParam_` (the
*output* extent).

`out_dim` as specified in `N_` and `dataStageParam_` corresponds to the **output** size — e.g. `128` with padding=1 and `126` without padding for a 128×128 input.

> [!WARNING]
> **`unneededPad_` is not the sum of the other two.** It is *not* padding-only.
> When the slack exceeds the available back padding, it is absorbed next from the
> **valid data region**, and that portion is counted in `unneededPad_` but in
> **neither** `unneededPadFront_` nor `unneededPadBack_`. In general:
>
> ```
> unneededPad_  >=  unneededPadFront_ + unneededPadBack_
> ```
>
> Equality holds **iff `unneededPad_ <= padBack`** (the *requested* back padding),
> because bucket 1 of the absorption loop below can absorb at most `padBack`
> elements before it starts consuming real data. So the strict inequality — and
> the mis-sizing risk for a consumer that assumes the sum — applies exactly when
> the slack exceeds the available back padding, which includes the common
> `padBack = 0` case whenever `unneededPad_ > 0`.

### Computing `unneededPad*` fields

The slack is charged against three buckets in a **fixed order** — back padding
first, then valid data, then front padding:

```
# on entry: padFront_/padBack_ hold the REQUESTED pads,
#           unneededPadFront_ = unneededPadBack_ = 0,
#           valid_size as defined above.
for (remaining = unneededPad_; remaining > 0; remaining--) {
  if      (padBack_   > 0) { padBack_--;  unneededPadBack_++;  }   // 1. back pad
  else if (valid_size > 0) { valid_size--;                     }   // 2. real data (uncounted)
  else if (padFront_  > 0) { padFront_--; unneededPadFront_++;  }   // 3. front pad
  else                       ERROR("Illegal sizing of window operation");
}
```

`valid_size` is a loop-local counter here; it is decremented to track how much
real data has been given up, but that count is never emitted — which is precisely
why `unneededPad_` can exceed the two front/back counters.

Two consequences a producer must respect:

1. **`padFront_`/`padBack_` are net values.** They are *decremented* as the
   unneeded counters are incremented, so the emitted `padFront_`/`padBack_` are
   the padding still **in use**, not the padding originally requested. The
   originally-requested back padding is `padBack_ + unneededPadBack_`.
2. **`unneededPadFront_` is rarely non-zero.** Bucket 3 is reached exactly when
   `unneededPad_ > padBack + valid_size` — i.e. back padding *and the entire
   valid region* are both exhausted first. That only happens in
   degenerate/over-allocated cases (e.g. `valid_size = 1` with `padFront >= 2`);
   for ordinary convolution and pooling it stays `0`.

### Worked examples

In every row `out` is the standard output extent
`floor((valid + padF + padB - k) / s) + 1`.

| Case | `valid` | `padF`/`padB` requested | k | s | `out` | allocated | consumed | `unneededPad_` | `uPF` | `uPB` | emitted `padF`/`padB` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 3×3 s1 p1 on 128 — exact tiling | 128 | 1 / 1 | 3 | 1 | 128 | 130 | 130 | **0** | 0 | 0 | 1 / 1 |
| 3×3 s2 p1 on 128 | 128 | 1 / 1 | 3 | 2 | 64 | 130 | 129 | **1** | 0 | 1 | 1 / 0 |
| 3×3 s3 p1 on 8 | 8 | 1 / 1 | 3 | 3 | 3 | 10 | 9 | **1** | 0 | 1 | 1 / 0 |
| 2×2 s2 on 7, no pad | 7 | 0 / 0 | 2 | 2 | 3 | 7 | 6 | **1** | 0 | 0 | 0 / 0 |
| 4×4 s6 on 8, no pad | 8 | 0 / 0 | 4 | 6 | 1 | 8 | 4 | **4** | 0 | 0 | 0 / 0 |

The rows split cleanly along the `unneededPad_ > padBack` condition above:

- **Rows 1–3 have `padBack = 1 >= unneededPad_`,** so all slack is absorbed by
  back padding and the sum invariant *holds* (`uPB` accounts for it fully). Note
  row 2: stride 2 over a 130-element span leaves exactly one element unread, and
  the requested back pad covers it — so the emitted `padBack_` drops to `0`.
- **Rows 4–5 have `padBack = 0 < unneededPad_`,** so the slack comes *entirely*
  out of valid data. Both front/back counters stay `0` and `unneededPad_` is the
  only field recording it — this is the case a consumer assuming the sum will
  mis-size. Row 5 (`k=4 s=6` on 8: span 4 of 8, `unneededPad_ = 4`) is the
  stride-exceeds-kernel shape.

---

| [← Previous: FoldManager](foldmanager.md) | [↑ Table of Contents](README.md) | [Next: Stick-Alignment Padding →](stick-padding.md) |
|:--|:--:|--:|
