# Padding

For window/padded operations, such as convolution, padding information should be added to both `N_` and `dataStageParam_` in `sdsc.dscs_[0]`,
capturing information about front/back padding, stride, and related kernel dimension. If a padded dimension is chunked across cores,
front/back padding should be set to `-1` in core datastage.

When a dimension is padded due to window/padded operations like convolution, details of padding need to be specified via the following fields.

## (i) paddingSizes_

```
"paddingSizes_": {
   "<padded dim>": {
          "padFront_": 0 or 1,
          "padBack_":  0 or 1,
          "totalSize_": total size inclusive of padding,
          "stride_": <stride applied with the op>,
          "dilation_": <dilation applied with the op>,
          "windowDim_": <window dim> e.g. "ki_"
       }
```

One set of entries need to be added for each dimension that is padded within the `paddingSizes_` structure. `paddingSizes_` itself
would be a sub-field of both `dcs[0].N_` and `dcs[0].dataStageParam_`. Regardless of whether padding is applied or not in a convolution
operation, `totalSize_` will correspond to the input size after padding. E.g. for an image of size 128×128, `totalSize_` would be 130
with a padding of 1 and 128 without padding. The dimension's size (as specified in `N_`, `dataStageParam_` per se.) will correspond
to the output image size and hence 128 with padding=1 and 126 without padding.

## (ii) padding_ sub-field in scheduleTree_

The `padding_` sub-field in `scheduleTree_` can take on the following values:

```
NOPAD
LOWERED_PADDED
PADDED_NOZEROPAD
PADDED_WZEROPAD
PADDED_FULLSPAN
PADDED_FULLSPAN_WUNNEEDED
```

**`PADDED_NOZEROPAD`** — to be used with conv2d when padding is non-zero.  
**`PADDED_FULLSPAN_WUNNEEDED`** — to be used with conv2d when padding is zero.

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: FoldProperty →](foldproperty.md) |
|:--|:--:|--:|
