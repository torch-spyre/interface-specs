# InductorKTIR: Replacing SDSC Codegen with KTIR in torch-spyre

---

## 1. Overview

- [2. Goals and Constraints](#2-goals-and-constraints)
- [3. Current SDSC Architecture](#3-current-sdsc-architecture)
  - [3.1 Registration and Activation](#31-registration-and-activation)
  - [3.2 Compilation Pipeline](#32-compilation-pipeline)
  - [3.3 Key Data Structures](#33-key-data-structures)
  - [3.4 Concretization](#34-concretization)
- [4. Alternative Analysis: Triton via KTIR](#4-alternative-analysis-triton-via-ktir)
  - [4.1 What the Path Would Entail](#41-what-the-path-would-entail)
  - [4.2 Why It Seems Attractive](#42-why-it-seems-attractive)
  - [4.3 Structural Incompatibilities](#43-structural-incompatibilities)
  - [4.4 Conclusion](#44-conclusion)
- [5. Recommended Architecture: Direct LoopLevel IR → KTIR](#5-recommended-architecture-direct-looplevel-ir--ktir)
  - [5.1 Integration Point](#51-integration-point)
  - [5.2 MLIR Module Structure](#52-mlir-module-structure)
  - [5.3 IR Mapping: OpSpec → KTIR](#53-ir-mapping-opspec--ktir)
  - [5.4 MLIR Emission Strategy](#54-mlir-emission-strategy)
  - [5.5 Dynamic Shapes](#55-dynamic-shapes)
  - [5.6 Dual-Backend Coexistence](#56-dual-backend-coexistence)
- [6. Key Design Decisions](#6-key-design-decisions)
- [7. Handling Unsupported Operations and Triton Kernels](#7-handling-unsupported-operations-and-triton-kernels)
  - [7.1 How Unsupported Ops Are Handled Today](#71-how-unsupported-ops-are-handled-today)
  - [7.2 A Triton-for-Spyre Kernel in the KTIR Path](#72-a-triton-for-spyre-kernel-in-the-ktir-path)
- [8. Migration Path](#8-migration-path)
- [9. Out of Scope](#9-out-of-scope)

---

torch-spyre's Inductor backend currently serializes compiled graphs to SDSC JSON bundles, which the closed-source DeepTools backend compiler ingests. SDSC JSON is an ad-hoc format with limited extensibility and no mechanism for expressing dynamic shapes or structured polyhedral access patterns in a compiler-interoperable way.

KTIR (`ktdp` MLIR dialect) is the replacement target format. It is an MLIR dialect with first-class support for Spyre's memory hierarchy, polyhedral coordinate tiles, and symbolic runtime arguments. This document describes how to replace the SDSC codegenerator in torch-spyre with one that emits KTIR, while keeping the existing `CustomPreSchedulingPasses` pipeline intact and allowing both backends to coexist during the transition.

---

## 2. Goals and Constraints

1. **`CustomPreSchedulingPasses` must be reusable without modification.** All passes in `passes.py` — `deadcode_elimination`, `propagate_spyre_tensor_layouts`, `coarse_tile`, `work_distribution`, `scratchpad_planning`, etc. — must run unchanged regardless of which codegen backend is selected.
2. **SDSC and KTIR backends coexist for at least one release.** A config flag selects which backend is active; both are registered and tested simultaneously.
3. **Inductor API surface is contained.** No new inductor internals are introduced beyond the current set: `register_backend_for_device`, `BaseScheduling`, and the `GraphLowering._update_scheduler` monkey-patch in `patches.py`.
4. **One MLIR module per compiled graph.** Semantically parallel to one SDSC bundle file.
5. **Config flag to select backend.** `spyre.ktir_backend` (bool) in `torch_spyre/_inductor/config.py`.
6. **Dynamic shapes support, staged.** Phase 1 concretizes symbolic sizes identically to SDSC. Phase 2 introduces `!ktdp.runtime_arg` for dynamic dimensions.
7. **Maintainability across PyTorch major releases.** The inductor API boundary does not grow.

---

## 3. Current SDSC Architecture

### 3.1 Registration and Activation

`torch_spyre/_inductor/__init__.py` registers the backend via:

```python
register_backend_for_device(
    DEVICE_NAME,
    SuperDSCScheduling,
    SpyrePythonWrapperCodegen,
    device_custom_config=config,
)
```

`patches.py:enable_spyre_context()` monkey-patches `GraphLowering._update_scheduler` to insert `CustomPreSchedulingPasses` before the scheduler object is constructed. This is the only hook into inductor internals that is not part of the public `register_backend_for_device` API.

### 3.2 Compilation Pipeline

```
Dynamo FX Graph
  → Inductor FX passes (CustomPreGradPasses, CustomPrePasses, CustomPostPasses)
  → LoopLevel IR lowering
  → GraphLowering._update_scheduler (monkey-patched)
      → CustomPreSchedulingPasses:
            deadcode_elimination
            propagate_spyre_tensor_layouts
            optimize_restickify_locations
            finalize_layouts
            insert_restickify
            insert_bmm_padding
            dedup_and_promote_constants
            chunk_large_tensors
            propagate_named_dims
            assign_dim_hints
            coarse_tile
            span_reduction
            work_distribution
            scratchpad_planning
      → Scheduler.__init__(graph.operations)
          → CustomPreFusionPasses: build_loop_scheduler_nodes, propagate_mutation_layouts
          → Scheduler.fuse_nodes()
          → CustomPostFusionPasses: memory_planning, spyre_fuse_nodes
  → scheduler.codegen() → SuperDSCScheduling.codegen_node() per node group
      → SpyreKernel → op_specs: list[OpSpec | UnimplementedOp | LoopSpec]
      → SpyreKernel.codegen_kernel()
          → compile_op_spec() → parse_op_spec() → SDSCSpec → generate_sdsc() → JSON dict
  → bundle.py:generate_bundle() → MLIR bundle file
  → SpyrePythonWrapperCodegen: async_compile.sdsc('kernel_name', ...)
```

### 3.3 Key Data Structures

**`OpSpec`** — one hardware operation:
- `op: str`, `is_reduction: bool`
- `iteration_space: dict[Symbol, tuple[Expr, int]]` — maps each iteration symbol to `(range_size, work_division)`
- `args: Sequence[TensorArg]`, `op_info: dict`
- `tiled_symbols: list[Symbol]` — symbols that advance per coarse-tile iteration

**`TensorArg`** — per-tensor access descriptor:
- `is_input`, `arg_index`
- `device_dtype: DataFormats`, `device_size: list[int]`
- `device_coordinates: list[Expr]` — sympy expressions describing per-element access
- `allocation: dict` — keys `"hbm"`, `"lx"`, `"pool"` → byte offset

**`LoopSpec`** — coarse-tile loop from `coarse_tile` pass:
- `count: Expr`, `body: list[OpSpec | UnimplementedOp | LoopSpec]`
- `tiled_symbols: list[Symbol]`

**`SDSCSpec`** — SDSC codegen intermediate:
- `opfunc`, `execution_unit` (`"pt"` for matmul, `"sfp"` otherwise)
- `iteration_space` (concretized), `num_cores`, `work_slices`
- `core_id_to_work_slice` — sympy expr mapping `core_id` → slice offset per dim
- `layouts: dict`, `args: list[SDSCArgs]`

**`SDSCArgs`** — per-tensor SDSC parameters:
- `layout`, `dim_order`, `data_format`
- `scales` — `1` = normal, `-1` = reduced, `-2` = stick-reduced
- `strides`, `offsets`, `max_dim_sizes`, `allocation`
- `start_address`, `backGap`

**`CountedLoopSchedulerNode`** — `FusedSchedulerNode` subclass wrapping ops that share a `loop_group_id`. `can_fuse()` returns `False` to prevent inductor's fusion from splitting loop groups.

### 3.4 Concretization

`_concretize_for_sdsc()` calls `V.graph.sizevars.size_hint()` on every symbolic size expression before `SDSCSpec` is constructed. This is the mechanism that limits SDSC to static shapes.

---

## 4. Alternative Analysis: Triton via KTIR

A separate project is building a Triton → KTIR compiler (a Triton backend that emits KTIR instead of PTX). The natural question is whether torch-spyre could reuse `TritonScheduling` — going LoopLevel IR → Triton source → Triton-to-KTIR → KTIR — rather than building a `KTIRScheduling` from scratch.

### 4.1 What the Path Would Entail

Concretely: replace `SuperDSCScheduling` with `TritonScheduling` (or a subclass of `SIMDScheduling`), allow `TritonKernel.codegen_kernel()` to produce Triton Python source, and pipe that source through the Triton-to-KTIR compiler at `async_compile` time. The wrapper would emit `async_compile.triton_via_ktir(...)` and the KTIR output would be handed to the backend.

### 4.2 Why It Seems Attractive

- `SIMDScheduling.can_fuse_vertical/horizontal` already handles the common fusion cases; we would not need to reimplement fusion logic.
- The Triton-to-KTIR compiler is an active project, potentially removing the need to write direct KTIR emission at all.
- Triton's tiling model is well-understood and `TritonKernel` is a stable, tested codebase.

### 4.3 Structural Incompatibilities

**Memory access model mismatch.** `TritonKernel` emits `tl.load(ptr + xoffset, mask=mask)` where `xoffset = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)`. This is a flat pointer-arithmetic model over a 1D/2D tiled iteration space. Spyre's model is fundamentally different: `TensorArg.device_coordinates` are sympy polyhedral expressions; `SDSCArgs.scales` encodes stick-dimension reductions; `allocation` maps tensors to HBM/LX/pool at known byte offsets; and `core_id_to_work_slice` maps each core to a specific region. There is no correspondence between `XBLOCK`/`RBLOCK` blocking and Spyre's explicit per-core work slices and stick alignment. The Triton-to-KTIR compiler would need to reconstruct this structured access information from flat pointer arithmetic — information that was explicit in the LoopLevel IR before `SIMDScheduling.group_fn` discarded it.

**`CustomPreSchedulingPasses` incompatibility.** The passes in `passes.py` write structured annotations onto LoopLevel IR nodes: `loop_info`, `dim_hints`, `stride_map`, `FixedTiledLayout`, and `allocation`. `SIMDScheduling.group_fn` collapses all iteration ranges to `(numel, rnumel)` pairs before any of the per-op codegen begins. This collapse discards the per-dimension structure that `assign_dim_hints`, `coarse_tile`, and `work_distribution` compute. The passes would run, but the annotations they produce would not survive into `SIMDKernel`'s tiling logic. Requirement 1 (passes must be reusable without modification) is violated not because the passes themselves change, but because the scheduling class no longer consumes their output.

**`CountedLoopSchedulerNode` / coarse-tile loops.** `coarse_tile` in `passes.py` produces `CountedLoopSchedulerNode` instances. `SIMDScheduling.can_fuse` is unaware of this node type. Plugging `CountedLoopSchedulerNode` into `SIMDScheduling`'s fusion logic would require modifying `SIMDScheduling` itself — the opposite of containment.

**Fusion model mismatch.** `SIMDScheduling` fuses nodes whose `(numel, rnumel)` groups are compatible. `spyre_fuse_nodes` (run in `CustomPostFusionPasses`) produces one bundle per graph based on Spyre-specific compatibility criteria — layout compatibility, memory bank conflicts, execution-unit homogeneity. Reconciling these two fusion models would require either overriding enough of `SIMDScheduling` that the reuse benefit disappears, or running `spyre_fuse_nodes` before handing nodes to `SIMDScheduling` and hoping the two models produce consistent groupings.

**Cross-layer translation loss.** Even if the above were resolved, Triton's output is Triton Python DSL source — a high-level text representation. The Triton-to-KTIR compiler must parse this and reconstruct polyhedral structure to emit `ktdp.construct_access_tile` ops. The LoopLevel IR already contains that structure. Routing through Triton deliberately discards and then attempts to reconstruct exactly what we already have.

**Maintenance.** The Triton path introduces two additional change points on every PyTorch major-version upgrade: `TritonScheduling`/`SIMDKernel` internals, and the Triton-to-KTIR compiler's compatibility with the Triton DSL version bundled with that PyTorch release.

### 4.4 Conclusion

The Triton path is not recommended. The modifications required to `SIMDScheduling`/`SIMDKernel` to accommodate `CountedLoopSchedulerNode`, `FixedTiledLayout`, and `core_id_to_work_slice` are extensive enough to negate the reuse benefit, while the path simultaneously discards the structured information that `CustomPreSchedulingPasses` produces. The direct LoopLevel IR → KTIR path preserves that structure end-to-end and is the right design.

The one scenario where this conclusion might be revisited: if the Triton-to-KTIR compiler matures to cover the full Spyre op set, and if a future architecture simplifies Spyre's memory model to the point where the flat pointer-arithmetic model and the polyhedral model converge. That is not the current situation; this is a ~90% confidence assessment, not a categorical rejection.

---

## 5. Recommended Architecture: Direct LoopLevel IR → KTIR

### 5.1 Integration Point

Two new classes in `torch_spyre/_inductor/`:

- **`KTIRScheduling(BaseScheduling)`** — in `ktir_scheduler.py` (or a new `ktir/` subpackage)
- **`KTIRPythonWrapperCodegen`** — in `ktir_wrapper.py`

`__init__.py:_autoload()` checks `config.ktir_backend` and calls `register_backend_for_device` with the appropriate pair:

```python
if config.ktir_backend:
    register_backend_for_device(
        DEVICE_NAME,
        KTIRScheduling,
        KTIRPythonWrapperCodegen,
        device_custom_config=config,
    )
else:
    register_backend_for_device(
        DEVICE_NAME,
        SuperDSCScheduling,
        SpyrePythonWrapperCodegen,
        device_custom_config=config,
    )
```

The `enable_spyre_context()` monkey-patch in `patches.py` is unchanged. `CustomPreSchedulingPasses` runs identically in both paths — this is the hard requirement, and it is satisfied by the integration point being downstream of the monkey-patch.

`KTIRScheduling` implements the same `BaseScheduling` interface as `SuperDSCScheduling`:

| Method | Behavior |
|---|---|
| `can_fuse_vertical` | `False` — Spyre manages fusion |
| `can_fuse_horizontal` | `False` |
| `codegen_node(node)` | Dispatches to `_codegen_counted_loop` or plain path; constructs `KTIRKernel` |
| `define_kernel(src_code, ...)` | Writes `async_compile.ktir(...)` into wrapper |

### 5.2 MLIR Module Structure

One `mlir.Module` per compiled graph, serialized to a `.mlir` file. The backend compiler reads this file directly, analogous to how it reads the SDSC bundle file today.

Structure:

```mlir
module @graph_<hash> {
  // Graph-level constants may appear as global memrefs

  func.func @kernel_<name>(%arg0: memref<?xf16, #ktdp.spyre_memory_space<HBM>>,
                            %arg1: memref<?xf16, #ktdp.spyre_memory_space<HBM>>,
                            %arg2: memref<?xf16, #ktdp.spyre_memory_space<HBM>>) {
    // body — see §5.3
    return
  }

  // Additional func entries for other kernel groups in the same graph
}
```

`CountedLoopSchedulerNode` groups produce `scf.for` nests within a `func.func`. Each `OpSpec` within a loop body maps to a sequence of `ktdp.*` ops. Multiple `FusedSchedulerNode` groups from `spyre_fuse_nodes` each produce their own `func.func` entry in the same module.

### 5.3 IR Mapping: OpSpec → KTIR

#### Memory views from `TensorArg`

`TensorArg.allocation` provides the base byte offset; `device_size` provides the shape; `device_coordinates` (sympy expressions) describe per-element access patterns.

```mlir
// TensorArg: allocation={"hbm": 0x1000}, device_size=[128, 64], dtype=f16
// → construct_memory_view with HBM memory space
%view = ktdp.construct_memory_view %base_addr,
    sizes: [128, 64], strides: [64, 1]
    { coordinate_set = affine_set<(d0, d1) : (d0 >= 0, d1 >= 0, d0 < 128, d1 < 64)>,
      memory_space = #ktdp.spyre_memory_space<HBM> }
    : memref<128x64xf16, #ktdp.spyre_memory_space<HBM>>
```

For LX (scratchpad) allocations:

```mlir
%lx_view = ktdp.construct_memory_view %lx_base,
    sizes: [32, 64], strides: [64, 1]
    { coordinate_set = affine_set<(d0, d1) : (...)>,
      memory_space = #ktdp.spyre_memory_space<LX> }
    : memref<32x64xf16, #ktdp.spyre_memory_space<LX>>
```

For distributed scratchpad tiles (from `scratchpad_planning`, multiple cores share a pool):

```mlir
%dist_view = ktdp.construct_distributed_memory_view
    (%view_core0, %view_core1 : memref<32x64xf16, ...>, memref<32x64xf16, ...>)
    : memref<64x64xf16, #ktdp.spyre_memory_space<LX>>
```

#### Per-core indexing

`SDSCSpec.core_id_to_work_slice` is a sympy expression mapping `core_id` to the slice offset for each dimension. In KTIR:

```mlir
%core_id = ktdp.get_compute_tile_id : index
// core_id_to_work_slice: offset = core_id * work_slice_size
%offset = affine.apply affine_map<(d0) -> (d0 * 32)>(%core_id)
```

#### Access tiles, loads, and stores

`TensorArg.device_coordinates` express per-element access as sympy affine maps over the iteration symbols and `core_id`. These translate to `ktdp.construct_access_tile`:

```mlir
// OpSpec: matmul output write, per-core work slice at [%i_offset, %j_offset]
%tile = ktdp.construct_access_tile %view[%i_offset, %j_offset]
    { access_tile_set  = affine_set<(d0, d1) : (d0 >= 0, d1 >= 0, d0 < 32, d1 < 64)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)> }
    : memref<128x64xf16, ...> -> !ktdp.access_tile<32x64xindex>

%result = ktdp.load %tile : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>
// ... compute ...
ktdp.store %output, %out_tile : tensor<32x64xf16>, !ktdp.access_tile<32x64xindex>
```

`SDSCArgs.scales` encodes stick-reduction dimensions (`-1`) and stick-reduced dimensions (`-2`). These map to `access_tile_set` constraints that restrict coordinate ranges along those dimensions.

#### Coarse-tile loops from `LoopSpec`

`LoopSpec.count` becomes an `scf.for` trip count; `tiled_symbols` advance the base address each iteration:

```mlir
// LoopSpec: count=4, tiled_symbol advances by tile_stride each iteration
%c0 = arith.constant 0 : index
%c4 = arith.constant 4 : index
%c1 = arith.constant 1 : index
scf.for %tile_idx = %c0 to %c4 step %c1 {
  %tile_base = affine.apply affine_map<(d0) -> (d0 * 128)>(%tile_idx)
  // nested OpSpec ops using %tile_base as base offset
}
```

#### Reductions

Ops with `OpSpec.is_reduction = True` use an accumulator pattern. The per-core partial result is computed, then reduced across participating cores using whatever cross-core reduction primitive the backend expects — the KTIR representation leaves this as a tagged `op_info` attribute on the kernel func.

### 5.4 MLIR Emission Strategy

Two options:

**Python API (`mlir_ktdp`):** Programmatic construction via `ktdp_d.ConstructMemoryViewOp(...)`, `ktdp_d.ConstructAccessTileOp(...)`, etc. Type-safe, verified at construction time. Same approach used in `ktir-mlir-frontend` tests.

**String generation:** Emit MLIR text directly as Python f-strings, then parse/verify. Simpler to implement initially; easier to read in `torch._dynamo.config.output_code = True` debug dumps. Mirrors how `TritonKernel.codegen_kernel()` returns Triton Python source and how `generate_sdsc()` / `generate_bundle()` assemble JSON text.

**Recommendation:** Start with string generation. Implementation velocity is higher; the IR structure is straightforward enough that string generation is not error-prone at this stage; and text dumps are an immediate debugging aid. Once the op mapping stabilizes and the KTIR op set stops changing rapidly, migrate to the Python API for correctness guarantees at construction time. A `--dump-ktir` flag (or `TORCH_SPYRE_DUMP_KTIR=1`) should emit the text form regardless of which emission path is active.

### 5.5 Dynamic Shapes

**Phase 1 (v1 / concurrent with SDSC replacement):** Call `V.graph.sizevars.size_hint()` on all symbolic size expressions before emitting `construct_memory_view` sizes and strides, exactly as `_concretize_for_sdsc()` does today. The emitted MLIR has fully static shapes.

**Phase 2 (coordinated with SDSC dynamic shape work):** Represent dynamic dimensions as `!ktdp.runtime_arg`:

```mlir
// Dynamic batch dimension known at runtime, upperbound=2048, granularity=128
%batch = ktdp.runtime_arg_extract value from %batch_sym
    : !ktdp.runtime_arg<index, granularity=128, upperbound=2048> -> index

%view = ktdp.construct_memory_view %base_addr,
    sizes: [%batch, 64], strides: [64, 1]
    { ... }
    : memref<?x64xf16, #ktdp.spyre_memory_space<HBM>>
```

The `granularity` and `upperbound` annotations are derived from `V.graph.sizevars` shape constraints and the `dim_hints` stamped by `assign_dim_hints`. Phase 2 is not planned for the initial KTIR release; it is noted here so the Phase 1 IR structure does not foreclose it.

### 5.6 Dual-Backend Coexistence

- `torch_spyre/_inductor/config.py` adds `ktir_backend: bool = False`
- `_autoload()` in `__init__.py` branches on `config.ktir_backend` (see §5.1)
- `CustomPreSchedulingPasses` is defined in `passes.py` and referenced by `patches.py`. No changes to either file.
- Test matrix: both backends run on the same model suite. SDSC output and KTIR output are validated independently against their respective backend compilers.
- The config flag may be set via `torch._inductor.config.spyre.ktir_backend = True` or via the env var `TORCH_SPYRE_KTIR_BACKEND=1` (consistent with existing `torch_spyre` config conventions).

---

## 6. Key Design Decisions

**Direct path over Triton path.** See §4. The Triton path discards the structured per-dimension information that `CustomPreSchedulingPasses` computes, requires invasive changes to `SIMDScheduling`, and adds a second compiler as a maintenance dependency. The direct path is a strict improvement on all axes.

**One module per graph, not per kernel.** The SDSC bundle is one file per graph; the KTIR module preserves that granularity. A per-graph module enables the backend compiler to see cross-kernel data flows (e.g. scratchpad aliasing between two `func.func` entries in the same graph). Per-kernel modules would require either cross-file linking at backend compile time or losing that visibility.

**`BaseScheduling` subclass, not monkey-patch.** `register_backend_for_device` already provides a clean extension point for the scheduling class. Monkey-patching `codegen_node` or `define_kernel` would make the two backends harder to test in isolation. The `GraphLowering._update_scheduler` monkey-patch in `patches.py` is retained only because there is no other hook for inserting pre-scheduling passes; a new monkey-patch is not introduced.

**Inductor API surface does not grow.** `KTIRScheduling` uses the same inductor entry points as `SuperDSCScheduling`. If inductor changes its `BaseScheduling` interface in a future PyTorch release, both backends are affected equally and the fix is in one place.

**String generation first.** The KTIR op set is still evolving. String generation avoids binding the implementation to nanobind API stability while the dialect stabilizes. Once the op mapping is proven and the dialect API is stable, migrating to `mlir_ktdp` Python API construction is a contained refactor inside `KTIRScheduling.codegen_node`.

**Config flag, not separate device type.** KTIR and SDSC share every component above the scheduling class: FX passes, LoopLevel lowering, `CustomPreFusionPasses`, `CustomPostFusionPasses`, and `CustomPreSchedulingPasses`. Introducing a separate device name (e.g. `"spyre_ktir"`) would require duplicating all of that registration machinery. A flag at `_autoload()` time is the minimal change.

The common prefix and the fork point are shown below:

```
Dynamo FX Graph
  → Inductor FX passes (CustomPreGradPasses, CustomPrePasses, CustomPostPasses)
  → LoopLevel IR lowering
  → GraphLowering._update_scheduler (monkey-patched)
      → CustomPreSchedulingPasses          ← shared, unchanged in both backends
      → Scheduler.__init__(graph.operations)
          → CustomPreFusionPasses          ← shared
          → Scheduler.fuse_nodes()         ← shared
          → CustomPostFusionPasses         ← shared
  → scheduler.codegen()
      │
      ├─ config.ktir_backend = False  ──────────────────────────────────────────
      │    SuperDSCScheduling.codegen_node()
      │      → SpyreKernel → OpSpec list
      │      → compile_op_spec() → SDSCSpec → JSON
      │    bundle.py:generate_bundle() → MLIR bundle file (.mlir)
      │    SpyrePythonWrapperCodegen: async_compile.sdsc(...)
      │    DeepTools backend compiler → device binary
      │
      └─ config.ktir_backend = True   ──────────────────────────────────────────
           KTIRScheduling.codegen_node()
             → SpyreKernel → OpSpec list   ← same SpyreKernel, same OpSpec
             → OpSpec → KTIR ops (construct_memory_view, construct_access_tile,
                                   load, store, scf.for, ...)
           MLIR module serialized to file (.mlir)
           KTIRPythonWrapperCodegen: async_compile.ktir(...)
           Backend compiler → device binary
```

The fork is entirely within `scheduler.codegen()`. Everything above that line — all FX passes, LoopLevel lowering, `CustomPreSchedulingPasses`, and the Scheduler construction including fusion — is identical in both paths. `SpyreKernel` and `OpSpec` are also shared; only the translation from `OpSpec` to the output format differs.

---

## 7. Handling Unsupported Operations and Triton Kernels

### 7.1 How Unsupported Ops Are Handled Today

There are no Triton kernels in the current SDSC path. Ops that Spyre cannot execute fall into two categories, both handled before `SuperDSCScheduling` is invoked:

**CPU fallbacks (`fallback_ops`).** Ops registered in `torch_spyre/ops/fallbacks.py` via `@register_fallback` have their inductor lowering suppressed (see `lowering.py:unregister_lowerings`). Inductor then lowers them as `ir.FallbackKernel` nodes, which become `ExternKernelSchedulerNode` instances in the scheduler. These are handled entirely by inductor's own wrapper machinery — `SuperDSCScheduling.codegen_node` never sees them. At runtime they execute eagerly on CPU, with tensor copies to/from Spyre device as needed.

**Unimplemented ops (`UnimplementedOp`).** Ops that reach `SpyreKernel` but cannot be lowered (e.g. an unsupported reduction type, an unexpected store value) produce an `UnimplementedOp` placeholder in `kernel.op_specs`. `SpyreAsyncCompile.sdsc()` calls `find_unimplemented()` on the spec list; if found, it returns a `SpyreUnimplementedRunner` that raises `RuntimeError` at runtime rather than failing silently at compile time.

### 7.2 A Triton-for-Spyre Kernel in the KTIR Path

A future scenario enabled by the Triton-to-KTIR compiler: a specific op (e.g. a hand-tuned attention kernel) is implemented as a Triton kernel targeting the Spyre device. Inductor's algorithm selection mechanism (`select_algorithm`) would register it as a `TritonTemplateCaller` for the `"spyre"` device. This is distinct from the CPU-fallback scenario — the kernel targets Spyre, not CPU, so it is routed through the Spyre backend.

**How inductor routes it.** The scheduler's `codegen()` loop dispatches each `SchedulerNode` to the backend registered for its device via `get_backend(node.get_device())`. Since the Triton-for-Spyre kernel targets device `"spyre"`, it is routed to `KTIRScheduling.codegen_node()` — not to `TritonScheduling`. `TritonScheduling` is never involved. The node arrives as a template node (`node.is_template()` returns `True`) carrying an `ir.TritonTemplateBuffer` rather than a `ComputedBuffer`.

**What `KTIRScheduling.codegen_node()` must do.** The current `codegen_node` path feeds `SchedulerNode` through `SpyreKernel` → `OpSpec` → KTIR ops. A template node has no `ComputedBuffer` and cannot go through that path. `KTIRScheduling.codegen_node()` needs to detect the template case and dispatch it differently:

```
KTIRScheduling.codegen_node(node):
  if node.is_template():
      → extract the Triton kernel source from the TritonTemplateBuffer
      → invoke the Triton-to-KTIR compiler on that source
      → incorporate the resulting KTIR func.func into the current module
  else:
      → SpyreKernel → OpSpec → KTIR ops  (existing path)
```

The Triton source and the directly-generated KTIR ops land in the same MLIR module, satisfying the one-module-per-graph constraint. The wrapper emits a single `async_compile.ktir(...)` call for the entire graph regardless of whether any nodes took the template path.

**Integration point for the Triton-to-KTIR compiler.** The Triton kernel source is available on `TritonTemplateBuffer` at codegen time. `KTIRScheduling` calls the Triton-to-KTIR compiler as a library or subprocess and receives back a `func.func` in KTIR text or as an `mlir_ktdp` IR object, which is inserted into the module alongside the ops generated by the `SpyreKernel` path. The exact API between `KTIRScheduling` and the Triton-to-KTIR compiler is a design decision for that compiler project; what matters here is that the boundary is a single call inside `KTIRScheduling.codegen_node()`, contained within the KTIR backend and invisible to `CustomPreSchedulingPasses` or any shared infrastructure.

**`UnimplementedOp` in the KTIR path.** If a node reaches `KTIRScheduling.codegen_node` but has no KTIR mapping and no template, `KTIRKernel` should produce an `UnimplementedOp` placeholder by the same convention `SpyreKernel` uses. `KTIRAsyncCompile.ktir()` calls `find_unimplemented()` and returns a `KTIRUnimplementedRunner` that raises `RuntimeError` at runtime, giving the same compile-time-safe / runtime-error semantics as the SDSC path.

---

## 8. Migration Path

**Phase 1 — Coexistence.** `KTIRScheduling` and `KTIRPythonWrapperCodegen` are implemented and gated behind `config.ktir_backend = False`. Both backends pass the model test suite. The KTIR backend is opt-in for early adopters and integration testing with the backend compiler.

**Phase 2 — Dynamic shapes.** `!ktdp.runtime_arg` support is added to `KTIRScheduling.codegen_node`. The SDSC dynamic shape work (developing in parallel) provides the `sizevars` constraint information needed to populate `granularity` and `upperbound`. KTIR becomes the recommended backend.

**Phase 3 — SDSC removal.** `SuperDSCScheduling`, `SpyrePythonWrapperCodegen`, `bundle.py:generate_bundle`, `generate_sdsc`, `SDSCSpec`, and `SDSCArgs` are removed. `config.ktir_backend` defaults to `True` and is then removed. `_autoload()` unconditionally registers the KTIR backend.

---

## 9. Out of Scope

- Backend compiler MLIR lowering pipeline (proprietary; not part of torch-spyre)
- Cross-kernel optimizations within the MLIR module (future backend compiler feature)
- Upstream contribution of `CustomPreSchedulingPasses` hook into PyTorch inductor
- Triton-to-KTIR compiler internals
- Quantization-aware codegen (separate project)
