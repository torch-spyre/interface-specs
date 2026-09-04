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

A symbolic address is a tensor's start address in device memory that is not a
concrete byte offset at compile time. The runtime value is supplied as an
operand to `sdscbundle.sdsc_execute` and bound to a symbol ID that appears as
a string placeholder in the JSON `startAddressCoreCorelet_` field.

### Symbolic Address — Runtime-Provided Base Address

**Use Case:** The caller supplies the tensor's base address at runtime (e.g.
the address of a model weight or activation buffer allocated by the runtime).
The MLIR function declares it as an `!sdscbundle.input_arg<index>` parameter,
extracts it, and passes it to `sdsc_execute` as the symbolic operand.

**MLIR:**

```mlir
module {
  func.func @symbolic_addr(%base_arg: !sdscbundle.input_arg<index>) {
    // Extract the runtime address from the bundle parameter
    %base = sdscbundle.input_arg_extract value from %base_arg
              : !sdscbundle.input_arg<index> -> index

    sdscbundle.sdsc_execute (%base) {
      sdsc_filename="op.json",
      symbol_ids=[-1]
    }
    return
  }
}
```

**Relevant SDSC JSON fields (`scheduleTree_` allocate node):**

```json
{
  "nodeType_": "allocate",
  "name_": "Tensor0",
  "ldsIdx_": 0,
  "component_": "hbm",
  "isStartAddrSymbolic_": true,
  "startAddressCoreCorelet_": {
    "dim_prop_func": ["Const"],
    "data_": {
      "[0, 0]": "-1"
    }
  }
}
```

The string `"-1"` in `data_` is the symbolic identifier. The backend matches it
to the operand bound to `symbol_ids=[-1]` in the MLIR and substitutes the
concrete runtime address just before job launch.

### Symbolic Address — Per-Core Addresses from a Runtime Base

**Use Case:** A 2-core operation where each core receives a different start
address derived from a caller-provided base. The per-core offsets are computed
using `affine.apply`.

**MLIR:**

```mlir
module {
  func.func @symbolic_addr_per_core(%base_arg: !sdscbundle.input_arg<index>) {
    %base = sdscbundle.input_arg_extract value from %base_arg
              : !sdscbundle.input_arg<index> -> index

    // Core 1 starts 128 bytes after core 0
    %offset = arith.constant 128 : index
    %addr_c1 = arith.addi %base, %offset : index

    sdscbundle.sdsc_execute (%base, %addr_c1) {
      sdsc_filename="op.json",
      symbol_ids=[-1, -2]
    }
    return
  }
}
```

**Relevant SDSC JSON fields (`scheduleTree_` allocate node):**

```json
{
  "nodeType_": "allocate",
  "name_": "Tensor0",
  "ldsIdx_": 0,
  "component_": "hbm",
  "isStartAddrSymbolic_": true,
  "startAddressCoreCorelet_": {
    "dim_prop_func": ["Map", "Const"],
    "data_": {
      "[0, 0]": "-1",
      "[1, 0]": "-2"
    }
  }
}
```

`"-1"` and `"-2"` are resolved to `%base` and `%addr_c1` respectively at launch
time.

---

## Symbolic Dimension Sizes

A symbolic dimension size is a tensor shape dimension — such as sequence length
or batch size — whose value is not fixed at compile time but varies per
invocation. The frontend declares the dimension as symbolic in the JSON and
supplies the runtime value via `symbol_ids` on `sdscbundle.sdsc_execute`, using
the same operand mechanism as symbolic addresses.

### Symbolic Dimension Size — Single Symbolic Batch Dimension

**Use Case:** A GELU operation over a batch whose size (`mb_`) is not known at
compile time. The JSON declares `mb_` as symbolic with a maximum of 32 and
granularity of 1. The MLIR passes the actual batch size as a runtime operand.

**MLIR:**

```mlir
module {
  func.func @symbolic_dim(%mb_arg: !sdscbundle.input_arg<index>,
                           %in_arg:  !sdscbundle.input_arg<index>,
                           %out_arg: !sdscbundle.input_arg<index>) {
    // Extract runtime values
    %mb  = sdscbundle.input_arg_extract value from %mb_arg
             : !sdscbundle.input_arg<index> -> index
    %in  = sdscbundle.input_arg_extract value from %in_arg
             : !sdscbundle.input_arg<index> -> index
    %out = sdscbundle.input_arg_extract value from %out_arg
             : !sdscbundle.input_arg<index> -> index

    // %mb  → symbol -1 (symbolic batch size)
    // %in  → symbol -2 (symbolic input  start address)
    // %out → symbol -3 (symbolic output start address)
    sdscbundle.sdsc_execute (%mb, %in, %out) {
      sdsc_filename="gelu.json",
      symbol_ids=[-1, -2, -3]
    }
    return
  }
}
```

**Relevant SDSC JSON fields:**

`DesignSpaceConfig.N_` declares the total dimensions; `dimToSymbolMapping_`
marks `mb_` as symbolic; `symbolicDimInfo_` constrains the valid runtime range:

```json
{
  "gelu": {
    "numCoresUsed_": 1,
    "coreIdsUsed_": [0],
    "N_": {
      "mb_": -1,
      "in_": 128
    },
    "dimToSymbolMapping_": {
      "mb_": ["-1"]
    },
    "dataStageParam_": {
      "0": {
        "ss_": {
          "mb_": -1,
          "in_": 128,
          "symbolicDimInfo_": {
            "mb_": { "maxSize_": 32, "granularity_": 1 }
          }
        },
        "el_": {
          "mb_": -1,
          "in_": 128,
          "symbolicDimInfo_": {
            "mb_": { "maxSize_": 32, "granularity_": 1 }
          }
        }
      }
    },
    "scheduleTree_": [
      {
        "nodeType_": "allocate",
        "name_": "gelu-Tensor0",
        "ldsIdx_": 0,
        "component_": "hbm",
        "isStartAddrSymbolic_": true,
        "startAddressCoreCorelet_": {
          "dim_prop_func": ["Const"],
          "data_": { "[0, 0]": "-2" }
        }
      },
      {
        "nodeType_": "allocate",
        "name_": "gelu-Tensor1",
        "ldsIdx_": 1,
        "component_": "hbm",
        "isStartAddrSymbolic_": true,
        "startAddressCoreCorelet_": {
          "dim_prop_func": ["Const"],
          "data_": { "[0, 0]": "-3" }
        }
      }
    ],
    "labeledDs_": [ "..." ],
    "computeOp_": [ "..." ]
  }
}
```

**Key points:**
- `mb_: -1` in `N_` signals an unset/symbolic dimension (the `-1` sentinel
  value, not a symbol ID).
- `dimToSymbolMapping_` links the dimension name `"mb_"` to symbol ID `"-1"`,
  which the backend matches to the first operand of `sdsc_execute`.
- `symbolicDimInfo_` in each `DataStageParam` provides the backend with
  `maxSize_: 32` (upper bound for memory planning) and `granularity_: 1`
  (runtime value must be a multiple of 1).
- Because this is a single-core operation the start addresses are symbolic but
  not dimension-derived — they are independently provided by the caller as
  symbol IDs `-2` and `-3`.

### Symbolic Dimension Size — Symbolic Dimension Split Across Cores

**Use Case:** A 2-core operation where the batch dimension is both symbolic
*and* split across cores. Each core's start address depends on the runtime
batch size, so `isStartAddrSymbolic_` must also be set and the per-core address
entries in `startAddressCoreCorelet_` are themselves symbolic identifiers.

**MLIR:**

```mlir
module {
  func.func @symbolic_dim_multicore(
      %mb_arg:    !sdscbundle.input_arg<index>,
      %in_c0_arg: !sdscbundle.input_arg<index>,
      %in_c1_arg: !sdscbundle.input_arg<index>,
      %out_c0_arg: !sdscbundle.input_arg<index>,
      %out_c1_arg: !sdscbundle.input_arg<index>) {

    %mb     = sdscbundle.input_arg_extract value from %mb_arg
                : !sdscbundle.input_arg<index> -> index
    %in_c0  = sdscbundle.input_arg_extract value from %in_c0_arg
                : !sdscbundle.input_arg<index> -> index
    %in_c1  = sdscbundle.input_arg_extract value from %in_c1_arg
                : !sdscbundle.input_arg<index> -> index
    %out_c0 = sdscbundle.input_arg_extract value from %out_c0_arg
                : !sdscbundle.input_arg<index> -> index
    %out_c1 = sdscbundle.input_arg_extract value from %out_c1_arg
                : !sdscbundle.input_arg<index> -> index

    // symbol -1: batch size
    // symbol -2/-3: per-core input  start addresses (core 0 and core 1)
    // symbol -4/-5: per-core output start addresses (core 0 and core 1)
    sdscbundle.sdsc_execute (%mb, %in_c0, %in_c1, %out_c0, %out_c1) {
      sdsc_filename="gelu_2core.json",
      symbol_ids=[-1, -2, -3, -4, -5]
    }
    return
  }
}
```

**Relevant SDSC JSON fields (`scheduleTree_` allocate nodes):**

```json
{
  "nodeType_": "allocate",
  "name_": "gelu-Input",
  "ldsIdx_": 0,
  "component_": "hbm",
  "isStartAddrSymbolic_": true,
  "startAddressCoreCorelet_": {
    "dim_prop_func": ["Map", "Const"],
    "data_": {
      "[0, 0]": "-2",
      "[1, 0]": "-3"
    }
  }
}
```

```json
{
  "nodeType_": "allocate",
  "name_": "gelu-Output",
  "ldsIdx_": 1,
  "component_": "hbm",
  "isStartAddrSymbolic_": true,
  "startAddressCoreCorelet_": {
    "dim_prop_func": ["Map", "Const"],
    "data_": {
      "[0, 0]": "-4",
      "[1, 0]": "-5"
    }
  }
}
```

**Key points:**
- Symbol `-1` is bound to the dimension size (`dimToSymbolMapping_`); symbols
  `-2` through `-5` are bound to per-core start addresses
  (`isStartAddrSymbolic_`). All five share the same `symbol_ids` operand list
  on the MLIR side.
- Both kinds of symbolic value appear together because the symbolic `mb_`
  dimension is split across two cores — the per-core address depends on the
  runtime batch size, so it cannot be a concrete integer in the JSON.
- The `granularity_` constraint on `mb_` ensures the runtime value is always
  evenly divisible by the number of cores (`granularity_` must be a multiple
  of the number of work slices in that dimension).

## Loop with Dynamic Addresses

**Use Case:** Iterative processing with address offsets — the same operation is applied repeatedly with a different memory address at each iteration.

**MLIR:**

```mlir
module {
  func.func @loop_ops() {
    %c0   = arith.constant 0    : index
    %c1   = arith.constant 1    : index
    %c8   = arith.constant 8    : index
    %base = arith.constant 1024 : index

    scf.for %i = %c0 to %c8 step %c1 {
      %addr = affine.apply affine_map<(d0)[s0] -> (s0 + 128*d0)> (%i)[%base]
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
    %c0   = arith.constant 0    : index
    %c1   = arith.constant 1    : index
    %base = arith.constant 1024 : index

    %addr_c0 = affine.apply affine_map<(d0)[s0] -> (s0 + 256*d0)> (%c0)[%base]
    %addr_c1 = affine.apply affine_map<(d0)[s0] -> (s0 + 256*d0)> (%c1)[%base]

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
  func.func @two_ops_with_intermediate(%in_arg:  !sdscbundle.input_arg<index>,
                                       %out_arg: !sdscbundle.input_arg<index>) {
    %in  = sdscbundle.input_arg_extract value from %in_arg
             : !sdscbundle.input_arg<index> -> index
    %out = sdscbundle.input_arg_extract value from %out_arg
             : !sdscbundle.input_arg<index> -> index

    // Allocate 32 KB for the intermediate tensor
    %inter = sdscbundle.device_mem_allocate 32768 bytes : index

    // First SDSC: reads from %in, writes result to %inter
    sdscbundle.sdsc_execute (%in, %inter) {
      sdsc_filename="sdsc_0_relu.json",
      symbol_ids=[-1, -2]
    }

    // Second SDSC: reads from %inter, writes final output to %out
    sdscbundle.sdsc_execute (%inter, %out) {
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
  func.func @pool_suballoc(%arg_0: !sdscbundle.input_arg<index>,
                           %arg_1: !sdscbundle.input_arg<index>,
                           %arg_2: !sdscbundle.input_arg<index>) {
    %a0 = sdscbundle.input_arg_extract value from %arg_0 : !sdscbundle.input_arg<index> -> index
    %a1 = sdscbundle.input_arg_extract value from %arg_1 : !sdscbundle.input_arg<index> -> index
    %a2 = sdscbundle.input_arg_extract value from %arg_2 : !sdscbundle.input_arg<index> -> index

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

    sdscbundle.sdsc_execute (%a0, %addr_0)                       {sdsc_filename="sdsc_0.json", symbol_ids=[-1, -2]}
    sdscbundle.sdsc_execute (%a1, %addr_16384)                   {sdsc_filename="sdsc_1.json", symbol_ids=[-3, -4]}
    sdscbundle.sdsc_execute (%addr_0, %addr_16384, %addr_32768)  {sdsc_filename="sdsc_2.json", symbol_ids=[-5, -6, -7]}
    sdscbundle.sdsc_execute (%addr_32768, %addr_49152, %a2)      {sdsc_filename="sdsc_3.json", symbol_ids=[-8, -9, -10]}

    return
  }
}
```

**Key points:**
- `device_mem_allocate` is placed in the entry block, outside any loop, so one buffer is reserved regardless of how many times surrounding control flow would pass through it.
- Each `arith.addi` computes an absolute device address by adding a constant offset to the pool base; alignment must be satisfied by the choice of offset.
- The frontend owns the layout: it must ensure that sub-ranges whose live ranges overlap are given disjoint offset intervals.

---

| [← Previous: MLIR Bundle API](MLIR-bundle-API.md) | [↑ Table of Contents](README.md) | [Next: Complete Example →](MLIR-complete-example.md) |
|:--|:--:|--:|
