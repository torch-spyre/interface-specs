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

## Device Memory Allocation (Intermediate Buffers)

**Use Case:** A kernel requires one or more intermediate tensors — buffers that hold outputs from one SDSC that become inputs to the next, or scratch space used only inside a single SDSC. These addresses are not supplied by the caller; instead the frontend requests device memory directly from the backend via `sdscbundle.device_mem_allocate`.

### Simple Intermediate Buffer

A single intermediate buffer is allocated and its base address is passed as the output address of the first SDSC and the input address of the second.

**MLIR:**

```mlir
module {
  func.func @two_ops_with_intermediate() {
    // Allocate 32 KB for the intermediate tensor
    %inter = sdscbundle.device_mem_allocate 32768 bytes : index

    // First SDSC: reads from %arg_0, writes result to %inter
    sdscbundle.sdsc_execute (%arg_0, %inter) {
      sdsc_filename="sdsc_0_relu.json",
      symbol_ids=[-1, -2]
    }

    // Second SDSC: reads from %inter, writes final output to %arg_1
    sdscbundle.sdsc_execute (%inter, %arg_1) {
      sdsc_filename="sdsc_1_layernorm.json",
      symbol_ids=[-3, -4]
    }

    return
  }
}
```

### Pool Sub-Allocation

When multiple intermediate tensors have non-overlapping live ranges, the frontend can allocate a single pool and sub-allocate it manually to keep total device memory within budget.

This example allocates a 64 KB pool and splits it into four 16 KB sub-buffers. The first two buffers are reused as inputs to `sdsc_2` after being written by `sdsc_0` and `sdsc_1` respectively, while the fourth is used as private scratch space by `sdsc_3`.

**MLIR:**

```mlir
module {
  func.func @pool_suballoc(%arg_0: index, %arg_1: index, %arg_2: index) {
    // Single pool covering all four intermediate regions
    %pool = sdscbundle.device_mem_allocate 65536 bytes : index

    // Sub-buffer offsets (one per 16 KB slot)
    %off_0     = arith.constant 0     : index
    %off_16384 = arith.constant 16384 : index
    %off_32768 = arith.constant 32768 : index
    %off_49152 = arith.constant 49152 : index

    // Derive absolute device addresses for each slot
    %addr_0     = arith.addi %pool, %off_0     : index   // sdsc_0 output → sdsc_2 input A
    %addr_16384 = arith.addi %pool, %off_16384 : index   // sdsc_1 output → sdsc_2 input B
    %addr_32768 = arith.addi %pool, %off_32768 : index   // sdsc_2 output → sdsc_3 input
    %addr_49152 = arith.addi %pool, %off_49152 : index   // sdsc_3 scratch (private)

    sdscbundle.sdsc_execute (%arg_0, %addr_0)                       {sdsc_filename="sdsc_0.json", symbol_ids=[-1, -2]}
    sdscbundle.sdsc_execute (%arg_1, %addr_16384)                   {sdsc_filename="sdsc_1.json", symbol_ids=[-3, -4]}
    sdscbundle.sdsc_execute (%addr_0, %addr_16384, %addr_32768)     {sdsc_filename="sdsc_2.json", symbol_ids=[-5, -6, -7]}
    sdscbundle.sdsc_execute (%addr_32768, %addr_49152, %arg_2)      {sdsc_filename="sdsc_3.json", symbol_ids=[-8, -9, -10]}

    return
  }
}
```

**Key points:**
- `device_mem_allocate` is placed in the entry block, outside any loop, so one buffer is reserved regardless of how many times surrounding control flow would pass through it.
- Each `arith.addi` computes an absolute device address by adding a constant offset to the pool base; alignment must be satisfied by the choice of offset.
- The frontend owns the layout: it must ensure that sub-ranges whose live ranges overlap are given disjoint offset intervals.

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: Complete Example →](MLIR-complete-example.md) |
|:--|:--:|--:|
