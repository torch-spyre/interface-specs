# Bundle Usage Examples

## Single Operation (No Symbols)

**Use Case:** Execute a single standalone operation with no dynamic address substitution.

**MLIR:**

```mlir
module {
  func.func @single_op() {
    sdscbundle.sdsc_execute () {sdsc_filename="gelu.json"}
    return
  }
}
```

## Sequential Operations (Kernel Fusion)

**Use Case:** Complex kernel decomposition — multiple operations chained in sequence within the same function (e.g., softmax).

**MLIR:**

```mlir
module {
  func.func @softmax() {
    sdscbundle.sdsc_execute () {sdsc_filename="max.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sub.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="exp.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sum.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="reciprocal.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="mul.json"}
    return
  }
}
```

## Symbolic Addresses

**Use Case:** Dynamic address calculation per core — start addresses are not known at compile time and are passed as symbolic operands resolved at runtime.

**MLIR:**

```mlir
module {
  func.func @symbolic_addr() {
    %addr_c0 = arith.constant 1024 : index
    %addr_c1 = arith.constant 1152 : index
    
    sdscbundle.sdsc_execute (%addr_c0, %addr_c1) {
      sdsc_filename="op.json",
      symbol_ids=[-1, -2]
    }
    return
  }
}
```

**SDSC JSON:**

```json
{
  "startAddressCoreCorelet_": {
    "data_": {
      "[0, 0]": "-1",
      "[1, 0]": "-2"
    }
  },
  "isStartAddrSymbolic_": true
}
```

## Loop with Dynamic Addresses

**Use Case:** Iterative processing with address offsets — the same operation is applied repeatedly with a different memory address at each iteration.

**MLIR:**

```mlir
module {
  func.func @loop_ops() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %base = arith.constant 1024 : index
    
    #addr_map = affine_map<(d0)[base] -> (base + 128*d0)>
    
    scf.for %i = %c0 to %c8 step %c1 {
      %addr = affine.apply #addr_map (%i)[%base]
      sdscbundle.sdsc_execute (%addr) {
        sdsc_filename="op.json",
        symbol_ids=[-1]
      }
    }
    return
  }
}
```

## Multi-Core with Per-Core Addresses

**Use Case:** Work division across cores — each core receives a different start address derived from its core ID, enabling parallel processing of distinct memory regions.

**MLIR:**

```mlir
module {
  func.func @multi_core() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %base = arith.constant 1024 : index
    
    #addr_map = affine_map<(core)[base] -> (base + 256*core)>
    
    %addr_c0 = affine.apply #addr_map (%c0)[%base]
    %addr_c1 = affine.apply #addr_map (%c1)[%base]
    
    sdscbundle.sdsc_execute (%addr_c0, %addr_c1) {
      sdsc_filename="op.json",
      symbol_ids=[-1, -2]
    }
    return
  }
}
```

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: Complete Example →](MLIR-complete-example.md) |
|:--|:--:|--:|
