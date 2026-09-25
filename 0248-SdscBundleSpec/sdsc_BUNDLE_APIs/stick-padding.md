# Stick-Alignment Padding (layout padding)

Every device tensor's **stick dimension** must be a whole multiple of the stick
size.  When a logical extent is not, the producer widens it.  This padding is
**independent of any operation** — it follows from the tensor layout alone — and
it is therefore *not* the same thing as the window padding described in
[padding.md](padding.md).

The central fact:

> **Stick-alignment padding is NOT described in the SDSC.**  The descriptor
> carries the **padded** extent as if it were the real one, and the logical extent
> does not appear anywhere.  `paddingSizes_` stays empty and every `padding`
> field reads `nopad`.

### Worked example: the padding is invisible

An `8 x 100` fp16 tensor.  fp16 sticks hold 64 elements, and `100 % 64 = 36`, so
28 lanes of padding are added to reach `128 = 2` sticks.

The emitted descriptor for a pointwise `mul` on that tensor:

| Field path | Value |
|---|---|
| `N_.out_` | `128` (padded extent) |
| `dataStageParam_[0].ss_.paddingSizes_` | `{}` (empty) |
| `scheduleTree_[0].coordinates_.coordInfo.out.padding` | `"nopad"` |

Three observations:

* `out_` is **128**, the padded extent.
* **`100` occurs nowhere in the descriptor** (verified by grep: zero
  occurrences).
* `paddingSizes_` is empty and every `padding` field is `nopad`.

The alignment is applied by widening the SDSC iteration space **before** the
descriptor is built, so nothing downstream ever sees the unaligned extent.

### Restickify converts the widening into `backGap`

There is **one** case where stick-alignment padding does become a declared SDSC
quantity: a **restickify** op.  On a `restickify` op, stick-alignment widening is applied
to the iteration space for both the old and the new stick dim. A dim that is not given arg's
stick dimension — typically the input's old stick dim — can then have a device size smaller than
the widened iteration extent. The shortfall is emitted as a positive `backGapCore_` entry on that arg.
The stick dim's own widening stays undeclared.

---

| [← Previous: Padding](padding.md) | [↑ Table of Contents](README.md) | [Next: Stick Layout Constraints →](stick-layout-constraints.md) |
|:--|:--:|--:|
