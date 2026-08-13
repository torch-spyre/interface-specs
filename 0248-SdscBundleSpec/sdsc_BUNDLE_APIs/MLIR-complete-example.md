# Complete Example — MLIR (Softmax Implementation)

## File: softmax.mlir

```mlir
module {
  func.func @softmax_dynamic() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %batch_size = arith.constant 32 : index
    
    %input_base = arith.constant 0 : index
    %temp_base = arith.constant 65536 : index
    %output_base = arith.constant 131072 : index
    
    #input_map = affine_map<(d0)[base] -> (base + 2048*d0)>
    #temp_map = affine_map<(d0)[base] -> (base + 2048*d0)>
    #output_map = affine_map<(d0)[base] -> (base + 2048*d0)>
    
    scf.for %b = %c0 to %batch_size step %c1 {
      %in_addr = affine.apply #input_map (%b)[%input_base]
      %tmp_addr = affine.apply #temp_map (%b)[%temp_base]
      %out_addr = affine.apply #output_map (%b)[%output_base]
      
      // max reduction
      sdscbundle.sdsc_execute (%in_addr, %tmp_addr) {
        sdsc_filename="sdscMax.json",
        symbol_ids=[-1, -2]
      }
      
      // subtract max
      sdscbundle.sdsc_execute (%in_addr, %tmp_addr, %tmp_addr) {
        sdsc_filename="sdscSub.json",
        symbol_ids=[-1, -2, -3]
      }
      
      // exp
      sdscbundle.sdsc_execute (%tmp_addr, %tmp_addr) {
        sdsc_filename="sdscExp.json",
        symbol_ids=[-1, -2]
      }
      
      // sum reduction
      sdscbundle.sdsc_execute (%tmp_addr, %tmp_addr) {
        sdsc_filename="sdscSum.json",
        symbol_ids=[-1, -2]
      }
      
      // reciprocal
      sdscbundle.sdsc_execute (%tmp_addr, %tmp_addr) {
        sdsc_filename="sdscReciprocal.json",
        symbol_ids=[-1, -2]
      }
      
      // multiply
      sdscbundle.sdsc_execute (%tmp_addr, %tmp_addr, %out_addr) {
        sdsc_filename="sdscMul.json",
        symbol_ids=[-1, -2, -3]
      }
    }
    
    return
  }
}
```

This example demonstrates:

- Loop-based batch processing
- Dynamic address calculation
- Sequential operation chaining
- Symbol-based parameterization
- Temporary buffer management

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: SDSC JSON API →](SDSC-json-api.md) |
|:--|:--:|--:|
