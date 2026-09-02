# MLIR Bundle API

## Bundle Container

A SuperDSC Bundle is expressed as a standard MLIR `module` containing a single
`func.func`. The `module` is a top-level container; operations inside it cannot
capture values defined outside. The `func.func` declares the bundle entry point:
it has no return values and its body ends with `return`.

**Syntax:**

```mlir
module {
  func.func @bundle_name() {
    // Bundle operations
    return
  }
}
```

**Operands:** None — all inputs to the bundle are expressed as symbolic operands
inside individual `sdscbundle.sdsc_execute` calls.

**Attributes:**

- **@bundle_name**: Identifier for the callable function region.

**Returns:** None — `func.func` in SDSC Bundles always has no return values.

**Constraints:**

- Exactly one `func.func` per `module`.
- All `sdscbundle.device_mem_allocate` calls must appear in the entry block of
  `func.func`, outside any `scf.for` loop.

**Example:**

```mlir
module {
  func.func @gelu_forward() {
    sdscbundle.sdsc_execute () {sdsc_filename="sdscGelu.json"}
    return
  }
}
```

---

## sdscbundle Dialect Operations

These operations are defined by the `sdscbundle` dialect and are the primary
means by which a frontend compiler communicates with the Spyre backend.

### sdscbundle.sdsc_execute

**Description:**

Instantiates and executes a SuperDSC operation. Each call references one SDSC
JSON file. A single `func.func` body may contain one or more `sdsc_execute`
calls in sequence, with each call referencing a different JSON file. This is how
kernel fusion is expressed — multiple operations (e.g., the five steps of a
softmax) are chained as sequential invocations within the same function. Each
invocation optionally substitutes symbolic addresses or sizes with the SSA values
provided as operands.

**Syntax:**

```mlir
sdscbundle.sdsc_execute (%operand1, %operand2, ...) {
  sdsc_filename = "path/to/sdsc.json",
  symbol_ids = [id1, id2, ...]
}
```

**Operands:**

- **%operand1, %operand2, ...** (optional): SSA values for symbolic parameters.
  - Type: `index`
  - Order must match `symbol_ids` order.
  - Can be constants, affine expressions, loop iterators, or addresses returned
    by `sdscbundle.device_mem_allocate`.
  - Each `sdsc_execute` call has its own independent operand list.

**Attributes:**

- **sdsc_filename** (required): `string` — relative path to the SDSC JSON file
  for this invocation; one file per call. Path is relative to the MLIR file
  location.

- **symbol_ids** (optional): `array<int>` — list of symbol IDs used in the
  SDSC. Values are negative integers (e.g., `-1`, `-2`, `-3`). Each ID maps
  positionally to one operand and corresponds to a symbolic start address or
  size inside the JSON. Symbol IDs are scoped per invocation — the same ID
  (e.g., `-1`) in two different calls is independent.

**Returns:** None.

**Constraints:**

- The number of operands must equal the length of `symbol_ids`.
- Symbol IDs must be unique within a bundle (i.e., the same ID cannot be reused
  across different `sdsc_execute` calls unless both invocations assign the same
  value to that symbol).

**Examples:**

Minimal — single operation, no symbols:

```mlir
sdscbundle.sdsc_execute () {sdsc_filename="sdsc.json"}
```

With a symbolic address:

```mlir
%addr = arith.constant 1024 : index
sdscbundle.sdsc_execute (%addr) {
  sdsc_filename="sdsc.json",
  symbol_ids=[-1]
}
```

With multiple symbolic addresses (one per core):

```mlir
%addr_c0 = arith.constant 1024 : index
%addr_c1 = arith.constant 1152 : index
sdscbundle.sdsc_execute (%addr_c0, %addr_c1) {
  sdsc_filename="sdsc.json",
  symbol_ids=[-1, -2]
}
```

Sequential operations — multiple JSON files in one function (softmax kernel
fusion):

![Softmax equation](figures/softmax_equation.png)

*The softmax function: for each element x_i, divide its exponent by the sum of
exponents of all elements.*

```mlir
module {
  func.func @sdsc_bundle() {
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0_maxnonstick.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_1_sub.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_2_exp.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_3_sumnonstick.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_4_realdiv.json"}
    return
  }
}
```

---

### sdscbundle.device_mem_allocate

**Description:**

Allocates a contiguous range of device memory for use within the bundle. This
operation is intended for buffers that are neither kernel inputs nor outputs —
intermediate tensors passed between consecutive SDSCs, and scratch space consumed
internally by a single SDSC. The backend reserves the requested bytes before the
first SDSC in the bundle executes and holds them for the entire kernel lifetime;
there is no matching deallocate. Because the allocation is bundle-scoped, placing
it inside a loop still produces only one buffer, not one per iteration.

The returned base address is a device byte address in the same address space as
the start addresses used inside `sdsc.json`. It can therefore be passed directly
to `sdscbundle.sdsc_execute` as the value of a symbolic start address, or used
as the base for sub-allocation arithmetic via `arith.addi`.

**Syntax:**

```mlir
%result = sdscbundle.device_mem_allocate <size> bytes : index
```

**Operands:** None.

**Attributes:**

- **size** (required): `integer` — size of the requested buffer in bytes. Must
  be a positive compile-time constant; symbolic sizes are not supported. The
  maximum single request is approximately 15 GB; the underlying segment is 16 GB,
  of which 1 GB is reserved for backend-generated programs and correction tensors.

**Returns:** A single `index` SSA value — the device byte address of the first
byte of the allocated buffer. Contents are undefined at allocation.

**Constraints:**

- Must appear in the entry block of the bundle function, outside any `scf.for`
  loop.
- Each call allocates its own non-overlapping range. The total device memory
  required by a bundle is the sum of all its `device_mem_allocate` requests, and
  that sum must remain within the ~15 GB budget.
- A frontend that needs to reuse space across tensors with non-overlapping live
  ranges should issue a single large `device_mem_allocate` and sub-allocate it
  manually using `arith.addi` offsets. The frontend is responsible for ensuring
  non-overlapping, correctly aligned sub-ranges.

**Examples:**

Minimal — allocate a single 64 KB buffer and pass it as a symbolic address:

```mlir
%buf = sdscbundle.device_mem_allocate 65536 bytes : index
sdscbundle.sdsc_execute (%buf) {
  sdsc_filename="op.json",
  symbol_ids=[-1]
}
```

Sub-allocation — carve a 64 KB pool into four 16 KB buffers:

```mlir
%pool = sdscbundle.device_mem_allocate 65536 bytes : index

%off_0     = arith.constant 0     : index
%off_16384 = arith.constant 16384 : index
%off_32768 = arith.constant 32768 : index
%off_49152 = arith.constant 49152 : index

%addr_0     = arith.addi %pool, %off_0     : index
%addr_16384 = arith.addi %pool, %off_16384 : index
%addr_32768 = arith.addi %pool, %off_32768 : index
%addr_49152 = arith.addi %pool, %off_49152 : index

sdscbundle.sdsc_execute (%arg_0, %addr_0)                  {sdsc_filename="sdsc_0.json", symbol_ids=[-1, -2]}
sdscbundle.sdsc_execute (%arg_1, %addr_16384)              {sdsc_filename="sdsc_1.json", symbol_ids=[-3, -4]}
sdscbundle.sdsc_execute (%addr_0, %addr_16384, %addr_32768){sdsc_filename="sdsc_2.json", symbol_ids=[-5, -6, -7]}
sdscbundle.sdsc_execute (%addr_32768, %addr_49152, %arg_2) {sdsc_filename="sdsc_3.json", symbol_ids=[-8, -9, -10]}
```

---

## Supporting MLIR Operations

The following operations are part of standard upstream MLIR dialects. They are
documented here only in the context of their permitted use within SDSC Bundles.
For full specifications see the [MLIR dialect
documentation](https://mlir.llvm.org/docs/Dialects/).

### scf.for

**Description:**

Loop construct from the Structured Control Flow (SCF) dialect. Used to
iteratively execute one or more SDSC operations, typically to process batches or
to apply address offsets per iteration.

**Syntax:**

```mlir
scf.for %iterator = %lower_bound to %upper_bound step %step {
  // Loop body with SDSC executions
}
```

**Operands:**

- **%lower_bound**: Starting value, inclusive (type: `index`).
- **%upper_bound**: Ending value, exclusive (type: `index`).
- **%step**: Increment per iteration (type: `index`).
- **%iterator**: Loop induction variable (type: `index`) — produced by the op,
  available inside the loop body.

**Attributes:** None — bounds and step are SSA values, not attributes.

**Returns:** None — `scf.for` in SDSC Bundles carries no results.

**Constraints:**

- Loop bounds must be compile-time constants.
- Loop-carried variables are not supported.
- Only the induction variable may be used directly inside the loop body.
- `sdscbundle.device_mem_allocate` must not appear inside the loop body.

**Example:**

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c8 = arith.constant 8 : index

scf.for %i = %c0 to %c8 step %c1 {
  %addr = affine.apply affine_map<(d0) -> (1024 + 128*d0)> (%i)
  sdscbundle.sdsc_execute (%addr) {
    sdsc_filename="sdsc.json",
    symbol_ids=[-1]
  }
}
```

---

### arith.constant

**Description:**

Defines a compile-time constant SSA value. In the SDSC Bundle context this is
primarily used to define base addresses, loop bounds, loop step values, and
sub-allocation offsets.

**Syntax:**

```mlir
%name = arith.constant <value> : <type>
```

**Operands:** None — the value is encoded as a compile-time attribute.

**Attributes:**

- **value** (required): The compile-time constant to be materialised; encoded as
  an integer literal in the source text. Always `index` type in SDSC Bundle
  usage.

**Returns:** A single SSA value of the declared type.

**Constraints:**

- Type must be `index` in SDSC Bundle usage.

**Example:**

```mlir
%base_addr   = arith.constant 1024 : index
%lower_bound = arith.constant 0    : index
%upper_bound = arith.constant 8    : index
%step        = arith.constant 1    : index
```

---

### arith.addi

**Description:**

Performs integer addition of two SSA values. In the SDSC Bundle context this is
used to compute memory addresses by adding a constant offset to a base address —
for example, deriving sub-buffer addresses from a pool base returned by
`sdscbundle.device_mem_allocate`.

**Syntax:**

```mlir
%result = arith.addi %lhs, %rhs : <type>
```

**Operands:**

- **%lhs**: Left-hand operand (type: `index`).
- **%rhs**: Right-hand operand (type: `index`).

**Attributes:** None — both operands are SSA values.

**Returns:** A single SSA value of the same type as the operands, containing the
integer sum.

**Constraints:**

- Both operands must have the same type (`index` in SDSC Bundle usage).

**Examples:**

Add a fixed offset to a base address:

```mlir
%base   = arith.constant 1024 : index
%offset = arith.constant 256  : index
%addr   = arith.addi %base, %offset : index
```

Combine a base address with a loop-derived stride:

```mlir
%base     = arith.constant 1024 : index
%stride   = arith.constant 128  : index
%i_stride = arith.muli %i, %stride : index
%addr     = arith.addi %base, %i_stride : index
sdscbundle.sdsc_execute (%addr) {
  sdsc_filename="sdsc.json",
  symbol_ids=[-1]
}
```

---

### affine.apply

**Description:**

Applies a compile-time affine map to a set of dimension and symbol operands,
producing a single `index` result. Used in SDSC Bundles to compute per-iteration
or per-core memory addresses from a base address and loop variables.

**Syntax:**

```mlir
%result = affine.apply affine_map<(dims)[symbols] -> (expression)> (dim_values)[symbol_values]
```

Affine maps may also be defined as top-level named aliases:

```mlir
#map_name = affine_map<(d0, d1, ...)[s0, s1, ...] -> (expression)>
```

**Operands:**

- **dim_values**: SSA values bound to dimension variables `d0, d1, ...`
  (e.g., loop iterators, core IDs).
- **symbol_values**: SSA values bound to symbol variables `s0, s1, ...`
  (e.g., base addresses — must be loop-invariant).

**Attributes:**

- **affine_map** (required): A compile-time affine expression. Supported
  operators:
  - Addition: `+`
  - Multiplication: `*`
  - Floor division: `floordiv`
  - Ceiling division: `ceildiv`
  - Modulo: `mod`

**Returns:** A single SSA `index` value — the result of evaluating the affine
expression with the supplied operands.

**Constraints:**

- The affine map must be expressible as a linear combination of its dimension
  and symbol variables; non-linear expressions are not permitted.
- Symbol variables must be loop-invariant (i.e., defined outside any enclosing
  `scf.for`).

**Examples:**

Single-variable address stride:

```mlir
#stride_map = affine_map<(d0)[base] -> (base + 128*d0)>
%addr = affine.apply #stride_map (%i)[%base_address]
```

Two-variable map combining iteration index and core ID:

```mlir
#addr_map = affine_map<(d0, core)[base] -> (base + 1024*d0 + 256*core)>
%addr = affine.apply #addr_map (%i, %core_id)[%base_address]
```

Concrete evaluation of the above map:

Given:
- `%i = 3` (iteration index)
- `%core_id = 2` (core identifier)
- `%base_address = 0x10000` (65536 in decimal)

```
%addr = 65536 + 1024*3 + 256*2
      = 65536 + 3072 + 512
      = 69120 (0x10E00 in hex)
```

This calculates a memory address by:
1. Starting at base address (65536)
2. Adding iteration offset (1024 bytes per iteration × 3)
3. Adding core-specific offset (256 bytes per core × 2)

---

| [← Previous: Overview](Overview.md) | [↑ Table of Contents](README.md) | [Next: Bundle Usage Examples →](MLIR-bundle-usage-examples.md) |
|:--|:--:|--:|
