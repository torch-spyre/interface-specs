# Complete Example — MLIR (Softmax Implementation)

## File: softmax.mlir

```mlir
module {
  func.func @softmax_dynamic(%in_arg:  !sdscbundle.input_arg<index>,
                              %out_arg: !sdscbundle.input_arg<index>) {
    %in_base  = sdscbundle.input_arg_extract value from %in_arg
                  : !sdscbundle.input_arg<index> -> index
    %out_base = sdscbundle.input_arg_extract value from %out_arg
                  : !sdscbundle.input_arg<index> -> index

    // Allocate a per-element temporary buffer (32 batches × 2048 bytes each)
    %temp_pool = sdscbundle.device_mem_allocate 65536 bytes : index

    %c0         = arith.constant 0  : index
    %c1         = arith.constant 1  : index
    %batch_size = arith.constant 32 : index

    scf.for %b = %c0 to %batch_size step %c1 {
      %in_addr  = affine.apply affine_map<(d0)[s0] -> (s0 + 2048*d0)> (%b)[%in_base]
      %tmp_addr = affine.apply affine_map<(d0)[s0] -> (s0 + 2048*d0)> (%b)[%temp_pool]
      %out_addr = affine.apply affine_map<(d0)[s0] -> (s0 + 2048*d0)> (%b)[%out_base]

      // max reduction: input → temp
      sdscbundle.sdsc_execute (%in_addr, %tmp_addr) {
        sdsc_filename="sdscMax.json",
        symbol_ids=[-1, -2]
      }

      // subtract max: (input, temp) → temp
      sdscbundle.sdsc_execute (%in_addr, %tmp_addr, %tmp_addr) {
        sdsc_filename="sdscSub.json",
        symbol_ids=[-3, -4, -5]
      }

      // exp: temp → temp
      sdscbundle.sdsc_execute (%tmp_addr, %tmp_addr) {
        sdsc_filename="sdscExp.json",
        symbol_ids=[-6, -7]
      }

      // sum reduction: temp → temp
      sdscbundle.sdsc_execute (%tmp_addr, %tmp_addr) {
        sdsc_filename="sdscSum.json",
        symbol_ids=[-8, -9]
      }

      // reciprocal: temp → temp
      sdscbundle.sdsc_execute (%tmp_addr, %tmp_addr) {
        sdsc_filename="sdscReciprocal.json",
        symbol_ids=[-10, -11]
      }

      // multiply: (temp, temp) → output
      sdscbundle.sdsc_execute (%tmp_addr, %tmp_addr, %out_addr) {
        sdsc_filename="sdscMul.json",
        symbol_ids=[-12, -13, -14]
      }
    }

    return
  }
}
```

This example demonstrates:

- Loop-based batch processing
- Dynamic address calculation using inline `affine.apply`
- Sequential operation chaining
- Symbol-based parameterization with unique symbol IDs across all `sdsc_execute` calls
- Intermediate buffer management via `sdscbundle.device_mem_allocate`
- Runtime-provided input/output addresses via `!sdscbundle.input_arg<index>`

---

| [← Previous: Bundle Usage Examples](MLIR-bundle-usage-examples.md) | [↑ Table of Contents](README.md) | [Next: SDSC JSON API →](SDSC-json-api.md) |
|:--|:--:|--:|
