# Stick Layout Constraints

## Overview

Each operation category imposes constraints on stick composition, restricting which dimensions can be present in the stick. Tensors must be padded to meet these constraints. There are no constraints on tensor layout beyond the stick.

**Important:** Stick constraints can cause a ripple effect—a tensor may need padding even in its non-stick dimension if that dimension appears in the stick of another tensor feeding the same operation. This ensures dimension span consistency across all tensors.

## BatchMatmul

The BatchMatmul operation has 4 types of semantic dimensions:

- **reduction_dim**: Present in Input1, Input2, NOT in Output1 (reduced via dot-product)
- **generated_dim**: Present in Input2, Output1, NOT in Input1
- **preserved_dim**: Present in Input1, Output1, NOT in Input2 (up to 2 dimensions)
- **noreuse_dim**: Present in all tensors (up to 2 dimensions)

**Output Tensor:**

- Stick: `[generated_dim=64]`
- Format: Always SEN169_FP16

**Input1 Tensor:**

- SEN169_FP16: `[reduction_dim=64]`
- FP8/INT8: `[reduction_dim=128]`
- INT4: `[reduction_dim=16, preserved_dim=2, reduction_dim=8]` (256 elements total)

**Input2 Tensor:**

- SEN169_FP16: `[generated_dim=64]`
- FP8/INT8: `[reduction_dim=2, generated_dim=64]`
- INT4: `[reduction_dim=4, generated_dim=64]`

**Note:** Input2 must be padded along `reduction_dim` (even though it's not in its stick) because `reduction_dim` is part of Input1's stick. This ensures dimension span consistency.

## Convolution

Same as BatchMatmul with one difference for INT4 Input1 stick layout:

- INT4 Input1: `[reduction_dim=16, W=2, reduction_dim=8]`
  - Where `W` is the width in pixels according to NHWC notation

All other precision formats follow BatchMatmul constraints.

## Reduction Operations

**Stick Reductions (sum/max/min/mean/absmax/exx2):**

- Reduction dimension must be the **only** dimension in stick
- Same stick layout for input and output
- Output has `scale=-2` for reduced dimension (stick dimension)

**Non-Stick Reductions (sumnonstick/maxnonstick/minnonstick/meannonstick/absmaxnonstick):**

- Any number of non-reduction dimensions allowed in stick
- Same stick layout for input and output
- **Note:** No non-stick version exists for exx2

## Unary and Broadcast Operations

**Constraints:**

- Any stick layout is acceptable, but all inputs and outputs must have the same stick layout.
- If a stick dimension has broadcast in a tensor, all stick dimensions of that tensor must have broadcast.
- If a stick dimension has broadcast in all tensors (inputs and output), then its size in the SDSC must be set to the number of elements that one stick would have if that dimension actually existed.

## Scan Operations

**Top-K Operations:**

- Neither the reduction dimension nor k can be in the stick
- Any number of other dimensions can be in the stick

## LayerNorm and EXX2 Operations

**Operations:** `layernormscale`, `layernormnorm`, `exx2`

**Constraint:**

- Stick should only have the normalization dimension in it

## Pooling Operations

**Constraint:**

- Window dimensions not allowed in the stick
- Any number of other dimensions can be in the stick

## Quantization Operations

### Down-casting Operations

**Input Constraints:**

- Input must have only one dimension in stick:
  - SEN169_FP16: `[inpdim=64]`
  - FP32: `[inpdim=32]`

**Output Stick Layouts by Family:**

**wt family** (weight packing):

- Pack elements from multiple input sticks in dimension different from input stick dimension
- Alternating after every element
- INT8/FP8: `[otherdim=2, inpdim=64]`
- INT4: `[otherdim=4, inpdim=64]`

**mb family** (mini-batch packing):

- Pack elements from multiple input sticks in dimension different from input stick dimension
- Alternating every 8 elements
- Output stick has dimension inserted at slice level
- INT8/FP8: `[inpdim=8, otherdim=2, inpdim=8]`
- INT4: `[inpdim=16, otherdim=2, inpdim=8]`

**ch family** (channel packing):

- Pack elements from multiple input sticks in same dimension as input stick dimension
- Alternating every 8 elements
- Output stick has only one dimension with more elements
- INT8/FP8: `[inpdim=128]`
- INT4: Not applicable

**Detailed Packing Descriptions:**

- **csqint4wt**: Pack from 4 input sticks in different dimension, alternating after every element
- **csqint4**: Pack from 4 input sticks (first 2 in same dimension, then 2 groups across different dimension), alternating every 8 elements
- **csqint8ch**: Pack from 2 input sticks in same dimension, alternating every 8 elements
- **csqint8mb**: Pack from 2 input sticks in different dimension, alternating every 8 elements
- **csqint8wt**: Pack from 2 input sticks in different dimension, alternating after every element

### Up-casting Operations

**Constraint:**

- Both input and output must have the same single dimension in stick
- For every stick of input, multiple sticks will be produced:
  - SEN169_FP16 to FP32: 2 output sticks per input stick
  - FP8 to SEN169_FP16: 2 output sticks per input stick

## Stick Altering Data Shuffle

**Operation:** `ReStickifyOpHBM`

Restickify converts a tensor from one stick layout to another — used when the
graph contains a reshape or layout change that requires re-tiling the data
(e.g. after a `VirtualReshape`). The `HBM` suffix indicates that data flows
through HBM during the conversion.

**Constraints:**

- Input stick must contain elements from exactly **one** dimension (`d1`).
- Output stick must contain elements from exactly **one** dimension (`d2`).
- `d1` and `d2` may be any primary dimensions — there is no restriction on which dimensions are chosen.
- Only SEN169_FP16 precision is supported.

---

## Core Work Division Constraints

Any constituent dimension of an operation may be split across cores. The
constraints below apply to the work assigned per core and must be satisfied
regardless of how the split is determined.

### Data Tensors

| Constraint | Detail |
|---|---|
| **Stick-multiple alignment** | The per-core work extent in every dimension that is present in the stick must be a multiple of the stick size for that dimension. |
| **DDR address span** | The contiguous range of device memory a single core addresses for any given tensor must not exceed **256 MB**. This is the per-core addressable range limit, separate from the 2 MB on-core LX scratchpad. |

> **Note:** When tensors of different data types share a stick variable, the
> stick-alignment check uses the largest `elems_per_stick` value across those
> tensors. Code that assumes 64 fp16 elements per stick is fp16-specific; the
> general limit is 128 bytes per stick regardless of dtype.

### Index Tensors (indirect access)

The stick-multiple alignment constraint does **not** apply to index tensors
used for indirect access. Instead, for each dimension present in the stick of
an index tensor, the per-core work extent must satisfy one of the following:

- It spans an **integral number of sticks** — the extent is an exact multiple
  of the stick size for that dimension, or
- It spans **fewer than one full stick** — the extent is strictly less than the
  stick size for that dimension.

A partial extent that covers more than one stick but is not a whole multiple of
the stick size is not permitted.

### Reduction Operations with Multiple Reduction Dimensions

When an operation reduces across more than one dimension, **only one** of its
reduction dimensions may be split across cores. There is no restriction on
operations that reduce along a single dimension.

---

| [← Previous: Padding](padding.md) | [↑ Table of Contents](README.md) | [Next: DesignSpaceConfig →](designspaceconfig.md) |
|:--|:--:|--:|
