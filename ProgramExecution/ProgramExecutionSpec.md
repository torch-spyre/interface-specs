# Program Execution Pipeline

**Authors:**
* @JRosenkranz

## **Summary**

This RFC introduces a layered stream-based execution model for torch-spyre. The backend compiler (deeptools) produces a `SpyreCode` JSON per SDSC, which torch-spyre translates into a `JobPlan` — an ordered sequence of `RuntimeOperation` steps with resolved device addresses plus tiling metadata. `SpyreStream` (torch-spyre) accepts JobPlans, handles tiled execution transparently, and submits operations to `RuntimeStream` (flex) for hardware dispatch. flex never sees a JobPlan — only fully-populated RuntimeOperations.

## **Scope**

This RFC introduces: `SpyreStream`, `JobPlan`, `JobPlanStep`, `ProgramCorrectionStep`, `PrepareKernel`, `LaunchKernel`, tiled execution, and the SpyreCode → RuntimeOperation translation flow.

It depends on (and summarizes relevant parts of):
- RuntimeStream RFC — `RuntimeStream`, `RuntimeOperation` base class, `RuntimeOperationH2D`/`D2H`/`Compute`/`HostCallback`. See the internal documentation for full details.
- SpyreAllocator RFC — `FlexAllocator`, `CompositeAddress`, `LogicalAddress`, `AllocationDirective`. See the internal documentation for full details.
- [SpyreCode Spec](../0277-SpyreCode/0277-SpyreCodeSpec.md) — SpyreCode JSON format, Job Preparation/Execution Plans

Imported definitions are summarized here for readability but the linked RFCs are authoritative.

## **Motivation**

The current connection between torch-spyre and the flex runtime relies on graph execution, which carries significant overhead for what is fundamentally a kernel launch job. The graph-based approach requires constructing, optimizing, and traversing a graph structure even for single-kernel execution, adding complexity and latency that is unnecessary for the common case.

This RFC replaces that approach with a layered stream architecture:

* **SpyreStream** (torch-spyre) implements the PyTorch Stream interface. It accepts `JobPlan` containers, handles tiled execution transparently, and submits operations to RuntimeStream. All methods are asynchronous; within a stream, operations execute in FIFO order; across streams, no ordering is guaranteed.

* **RuntimeStream** (flex) is the hardware execution queue. It accepts `RuntimeOperation` objects via `launchOperation()`, maintains FIFO ordering, and dispatches to the Spyre driver. See the RuntimeStream RFC in the internal documentation for full details.

* **JobPlan** is a torch-spyre internal container bundling an ordered sequence of `RuntimeOperation` steps (with resolved `CompositeAddress` values) and tiling metadata. flex never sees a JobPlan.

The SpyreAllocator's VF mode also shapes this design: `FlexAllocator` returns `CompositeAddress` values directly — a descriptor containing one or more `LogicalAddress` chunks. torch-spyre stores the `CompositeAddress` in the `at::DataPtr` and passes it back to flex for deallocation and to SpyreStream for execution (see SpyreAllocator RFC in the internal documentation).

## **Proposed Implementation**

### Core Components

#### CompositeAddress

A descriptor that represents the device-side location of an allocation. A `CompositeAddress` contains one or more `LogicalAddress` chunks (`region_id` + `offset`) that together form the allocation. For simple (non-interleaved) allocations, it contains exactly one chunk; for NUMA-aware interleaved allocations, it contains one chunk per memory domain. `CompositeAddress` is returned by `FlexAllocator.allocate()` and crosses the flex → torch-spyre boundary — torch-spyre stores it in a `SharedOwnerCtx` within the `at::DataPtr` and passes it back to flex for deallocation and execution. See the SpyreAllocator RFC in the internal documentation for full details on `CompositeAddress` structure.

#### LogicalAddress

A reference to a location in device memory. LogicalAddress is the building block of `CompositeAddress` — each chunk in a `CompositeAddress` contains a `LogicalAddress`. torch-spyre can access `LogicalAddress` values through the `CompositeAddress` chunks (e.g., to extract offsets for program correction). Within flex, RuntimeStream and the ControlBlock segment table consume `LogicalAddress` directly. A LogicalAddress identifies only a location — size is stored separately alongside it.

Properties:

| Property | Description |
|----------|-------------|
| `region_id` | Identifies the memory region. In VF mode, this is an index into a firmware lookup table that maps to the physical address of the region. In PF mode, this is the physical address of the region itself. |
| `offset` | Byte offset of the allocation within the region. Always 128-byte aligned. |

The same structure is used in both PF and VF modes — only the interpretation of `region_id` differs. All flex-internal components (RuntimeStream) work uniformly with LogicalAddress regardless of mode.

#### SpyreAllocator / FlexAllocator

Manages device memory in two layers: `SpyreAllocator` (torch-spyre) is a thin wrapper implementing PyTorch's `at::Allocator` interface; `FlexAllocator` (flex) is the core memory manager. FlexAllocator manages a pool of up to 8 memory regions in both PF and VF modes, carving individual allocations from them as 128-byte-aligned blocks.

`FlexAllocator.allocate()` accepts an optional `AllocationDirective` that controls placement relative to memory domains (e.g., bind to a specific memory domain, or interleave across multiple domains). It returns a `CompositeAddress` directly — a descriptor containing one or more `LogicalAddress` chunks (`region_id` + `offset`). For simple allocations, the `CompositeAddress` has a single chunk; for interleaved allocations, one chunk per memory domain. torch-spyre stores the `CompositeAddress` in the `at::DataPtr` and passes it to SpyreStream for execution. A `DeviceTopology` interface exposes the device's memory domain structure (domain count, capacity, core affinity) so callers can construct informed directives.

See the SpyreAllocator RFC in the internal documentation for full details on allocation strategies, placement directives, memory region management, and memory lifecycle.

#### SpyreTensor

A tensor residing on-device. Carries metadata required for execution:

* **shape**: Logical dimensions (e.g., `[4096, 1024]`)
* **stride**: Memory stride per dimension (bytes between consecutive elements along each axis)
* **data_type**: Element type (e.g., float32, uint32)
* **composite_address**: A `CompositeAddress` identifying the tensor's allocation on device (from FlexAllocator)
* **offset**: Storage offset in bytes. Non-zero when the tensor is a view into a larger storage (e.g., created via slicing or `torch.narrow`)
* **size_bytes**: Total byte size of the tensor data
* **layout**: A `SpyreTensorLayout` describing the tensor's tiled layout on device — includes `device_size` (tiled dimensions on device), `device_dtype` (on-device data type), and `dim_map` (mapping between logical shape and device dimensions)

#### RuntimeOperation

The abstract base class for all operations submitted to `RuntimeStream`. This is the same concept as `RuntimeOperation` in the RuntimeStream RFC and corresponds to `JobPlanCommand` in the SpyreCode RFC (see [SpyreCode](#spyrecode) for the mapping). Each subclass is self-contained, carrying all the metadata it needs for execution.

See the RuntimeStream RFC in the internal documentation for the base `RuntimeOperation` class definition (pipeline barriers, completion callbacks) and the standard operation types (`RuntimeOperationH2D`, `RuntimeOperationD2H`, `RuntimeOperationHostCallback`, `RuntimeOperationCompute`). Additional operation types (e.g., firmware metrics collection, memory activation/deactivation, RDMA for inter-device communication, allocation operations) will be added as the `RuntimeOperation` hierarchy evolves. The hierarchy is extensible by design.

**Binary allocation:** During Job Preparation Plan execution, torch-spyre allocates space for each `RuntimeOperationCompute` step's binary via `SpyreAllocator` (which delegates to `FlexAllocator`, using an `AllocationDirective` hinting that this is a program binary so the runtime can place it in the appropriate memory region). The allocation is a single contiguous block whose size is specified by the backend compiler's allocation metadata. This block always holds the program binary, and conditionally includes space for program correction tensors (at compiler-specified offsets, if the kernel requires program correction) and intermediate data tensors the backend compiler needs for scheduling spillover (at compiler-specified offsets, if the backend compiler needed additional DDR space for scheduling). The resulting `CompositeAddress` is stored on the `RuntimeOperationCompute` step by SpyreStream. When constructing a ControlBlock for dispatch, RuntimeStream handles the segment table mapping internally — setting the appropriate segment table entry so that the program binary is addressable from the device's perspective. See the SpyreAllocator RFC in the internal documentation for full details on program allocation and segment mapping.

##### Program Correction

Program correction is a **torch-spyre concern** — flex is unaware of it. When a kernel uses symbolic addresses or shapes, the backend compiler produces a unified binary containing both a correction program and the compute program, along with correction metadata. At execution time, torch-spyre must run a host-side function to produce a correction tensor, transfer it to device, and then launch the compute.

This is expressed as a sequence of generic `RuntimeOperation` objects — flex sees a `RuntimeOperationHostCallback` followed by a `RuntimeOperationH2D` followed by a `RuntimeOperationCompute`, with no knowledge that the callback is performing program correction. The correction-specific metadata lives at the JobPlan level in a `ProgramCorrectionStep` (see [JobPlan](#jobplan) for how this is embedded in the step sequence).

```cpp
struct ProgramCorrectionStep {
    // The host function that performs program correction.
    HostComputeFunction function;

    // Metadata from the compiler (e.g., hcm.json / vdci.json) describing how input
    // arguments (symbols) must be interpreted in the context of the job binary —
    // for example, mapping a symbolic dimension size to a loop count correction.
    json program_correction_metadata;

    // Size of the correction tensor output buffer (from SpyreCode oshape).
    size_t output_buffer_size;
};
```

When SpyreStream processes a `JobPlanStep` that has an associated `ProgramCorrectionStep`, it constructs a closure capturing the correction function and metadata, then emits a plain `RuntimeOperationHostCallback` followed by the H2D and Compute:

```cpp
// In SpyreStream::Launch, when processing a step with program correction:
std::vector<std::unique_ptr<RuntimeOperation>> ops;
auto correction_buf = correction_buffers_.acquire(step.correction->output_buffer_size);
auto callback = [correction = *step.correction, tensors, correction_buf]() {
    correction.function(tensors, correction.program_correction_metadata, correction_buf);
};
ops.push_back(std::make_unique<RuntimeOperationHostCallback>(
    /*enablePipelineBarrier=*/true, callback, nullptr));
ops.push_back(std::make_unique<RuntimeOperationH2D>(correction_buf, device_addr, size));
ops.push_back(std::make_unique<RuntimeOperationCompute>(binary_addr));
```

**Program correction flow:**
1. The `RuntimeOperationHostCallback` executes the correction closure on the host CPU, taking the resolved symbol values (tensor virtual addresses, shape values) and the `program_correction_metadata` as input, producing a correction tensor in the pre-allocated buffer.
2. The subsequent `RuntimeOperationH2D` transfers this correction tensor to a reserved location on device (within the program allocation, at a compiler-specified offset after the program binary — see SpyreAllocator RFC in the internal documentation for details on program allocation layout).
3. The `RuntimeOperationCompute` launches the unified binary as a single compute CB — internally, the correction program reads the correction tensor to patch the compute program, then the compute program executes.

The runtime does not distinguish between the correction program and the compute program; they are opaque within the single CB.

> **RuntimeStream RFC dependency:** This design requires that a `RuntimeOperationHostCallback` operation is not considered complete until the callback function returns — so that FIFO ordering guarantees the subsequent `RuntimeOperationH2D` does not start until the correction tensor is written. The current RuntimeStream RFC language ("the callback does not block the stream") is ambiguous on this point and should be clarified. See unresolved question #9.

#### SpyreCode

`SpyreCode` is a JSON artifact produced by the backend compiler (deeptools) for each SDSC. It is the contract between the compiler and torch-spyre — deeptools writes it, torch-spyre reads it. See the [SpyreCode Spec](../0277-SpyreCode/0277-SpyreCodeSpec.md) for the authoritative definition.

SpyreCode contains two plans:

* **Job Preparation Plan** — Executed once per SDSC. Contains `Allocate` and `InitTransfer` commands that torch-spyre uses to allocate device memory (via `SpyreAllocator`) and load binaries (via `RuntimeOperationH2D`). The resulting `CompositeAddress` values are stored and used to populate the execution plan's operations.

* **Job Execution Plan** — Executed per invocation. Contains `ComputeOnHost`, `ComputeOnDevice`, and `DataTransfer` commands that torch-spyre translates into `RuntimeOperation` objects with resolved addresses.

**SpyreCode → RuntimeOperation mapping:**

| SpyreCode Command | RuntimeOperation | Notes |
|---|---|---|
| `Allocate` | _(not a RuntimeOperation)_ | torch-spyre calls `SpyreAllocator.allocate()` → `CompositeAddress` |
| `InitTransfer` | `RuntimeOperationH2D` | Binary loading — host file to device at allocated address |
| `ComputeOnHost` | `ProgramCorrectionStep` (JobPlan-level) → `RuntimeOperationHostCallback` | torch-spyre builds a closure from correction metadata; flex sees a plain HostCallback |
| `ComputeOnDevice` | `RuntimeOperationCompute` | `job_bin_ptr` resolved to `CompositeAddress` from preparation |
| `DataTransfer(direction=0)` | `RuntimeOperationH2D` | `dev_ptr` resolved to `CompositeAddress`; `host_handle` resolved to `void*` |
| `DataTransfer(direction=1)` | `RuntimeOperationD2H` | Same address resolution as above |

**Translation flow:**

1. `torch.compile` triggers inductor, which produces SDSC inputs for deeptools
2. Deeptools compiles each SDSC and produces a `SpyreCode` JSON
3. torch-spyre parses the SpyreCode and executes the **Job Preparation Plan**:
   a. `Allocate` → `SpyreAllocator.allocate()` (delegates to `FlexAllocator`) → `CompositeAddress`
   b. `InitTransfer` → constructs a `RuntimeOperationH2D` with the binary data and the allocated `CompositeAddress`, submits to `RuntimeStream`
4. torch-spyre translates the **Job Execution Plan** into a `JobPlan` — resolving the compiler's symbols to `CompositeAddress` values (via `LogicalAddress`) using the allocations from step 3, and storing the `expected_input_shapes` from the SpyreCode
5. The resulting `JobPlan` is cached by torch-spyre and submitted to `SpyreStream.Launch` on each invocation

> **SpyreCodeSpec dependencies:** This RFC assumes the following fields exist in the SpyreCode JSON that are not yet defined in [SpyreCodeSpec.md](../0277-SpyreCode/0277-SpyreCodeSpec.md):
> - `expected_input_shapes`: per-kernel compiled tile dimensions (list of shape lists)
> - Clarification of how `dev_ptr` virtual addresses map to `LogicalAddress(region_id, offset)`
> - Clarification of `host_handle` type/resolution
>
> These should be added to 0277-SpyreCodeSpec.md before implementation begins.

#### JobPlan

A `JobPlan` is a **torch-spyre internal** container that bundles everything needed to execute a unit of work on a stream. It is produced by translating a SpyreCode's Job Execution Plan (see [SpyreCode](#spyrecode)) after the Job Preparation Plan has been executed. flex never sees a JobPlan — SpyreStream extracts the operations and submits them to `RuntimeStream.launchOperation()` as a `vector<RuntimeOperation>`.

A JobPlan is self-contained: if a compute requires program correction, the correction callback, the correction tensor DMA, and the device compute are all separate steps in the same JobPlan. For pure data movement (e.g., tensor `.to(device)` or binary loading), a JobPlan with only DMA steps is used.

**Producers:**
* **Backend compiler (deeptools) via torch-spyre:** Deeptools produces a `SpyreCode` JSON per SDSC. torch-spyre translates the SpyreCode into a JobPlan — executing the Job Preparation Plan (allocations, binary loading) and translating the Job Execution Plan into `JobPlanStep` entries with resolved `CompositeAddress` values. A single `torch.compile` call may produce multiple SDSCs, resulting in multiple JobPlans.
* **Communications libraries:** Create JobPlans for inter-device data transfers, collective operations, or other multi-step communication patterns.
* **torch-spyre:** Assembles JobPlans for tensor `.to(device)` moves (single `RuntimeOperationH2D` step), tensor `.to("cpu")` readbacks (single `RuntimeOperationD2H` step), or any other sequence of operations it needs to containerize.

```cpp
struct JobPlanStep {
    // The RuntimeOperation to submit (H2D, D2H, Compute, HostCallback, etc.)
    // All CompositeAddress fields are resolved before construction.
    std::shared_ptr<RuntimeOperation> operation;

    // If present, this step's operation is a host callback performing program
    // correction. SpyreStream uses this metadata to construct the closure for
    // the RuntimeOperationHostCallback. Only meaningful when operation is a
    // RuntimeOperationHostCallback.
    std::optional<ProgramCorrectionStep> correction;
};

struct JobPlan {
    // Ordered sequence of steps. SpyreStream walks this sequence, constructing
    // RuntimeOperations (building closures for correction steps) and submitting
    // them to RuntimeStream.
    std::vector<JobPlanStep> steps;

    // Compiled tile dimensions from SpyreCode, one entry per kernel input tensor.
    // Used by SpyreStream for tiling detection.
    // Empty for pure DMA JobPlans (e.g., tensor .to(device)).
    std::vector<std::vector<int64_t>> expected_input_shapes;
};
```

> **SpyreCodeSpec dependency:** `expected_input_shapes` is not yet defined in [SpyreCodeSpec.md](../0277-SpyreCode/0277-SpyreCodeSpec.md). It must be added to the SpyreCode JSON spec before implementation. Until then, this RFC defines the contract expectation.

| Property | Description |
|----------|-------------|
| `JobPlanStep.operation` | A fully-populated `RuntimeOperation` (with resolved `CompositeAddress` values) |
| `JobPlanStep.correction` | Optional `ProgramCorrectionStep`. When present, SpyreStream builds a closure from the correction metadata and uses it as the callback for the `RuntimeOperationHostCallback` in `operation`. |
| `JobPlan.steps` | Ordered list of `JobPlanStep` entries |
| `JobPlan.expected_input_shapes` | Compiled tile dimensions from SpyreCode. Used by SpyreStream for tiling detection. Empty for pure DMA JobPlans. |

**Step examples** (all `CompositeAddress` fields resolved after preparation):

**Simple compute (no program correction):**
```
steps:
  [0] { operation: RuntimeOperationCompute(<binary allocation>), correction: nullopt }
expected_input_shapes: [[1024, 1024]]
```

**Compute with program correction:**
```
steps:
  [0] { operation: RuntimeOperationHostCallback(barrier=true),
         correction: { function=correct_fn, metadata=hcm.json, output_size=2048 } }
  [1] { operation: RuntimeOperationH2D(correction_buffer, <program alloc offset>, 2048),
         correction: nullopt }
  [2] { operation: RuntimeOperationCompute(<binary allocation>),
         correction: nullopt }
expected_input_shapes: [[1024, 1024]]
```

SpyreStream processes step [0] by building a closure from the `ProgramCorrectionStep` and setting it as the callback on the `RuntimeOperationHostCallback`. All three steps are then submitted to `RuntimeStream.launchOperation()` as a `vector<RuntimeOperation>`.

**Pure data transfer:**
```
steps:
  [0] { operation: RuntimeOperationH2D(tensor_data, <allocation>, 4096), correction: nullopt }
```

#### Ownership and Lifecycle

| Object | Owner | Lifecycle |
|--------|-------|-----------|
| JobPlan, RuntimeOperations | torch-spyre | Cached for reuse across invocations. torch-spyre is responsible for creation, caching, and destruction. |
| CompositeAddress mappings | flex (FlexAllocator) | Freed when torch-spyre calls `FlexAllocator.deallocate()`. Destruction of a JobPlan does **not** implicitly free device memory — deallocation must be explicit. |
| Binary on device | Tied to CompositeAddress | Deallocating the CompositeAddress frees the device memory. The on-disk binary (`binary_path`) is independent and may be deleted after loading without affecting the device copy. |
| SpyreTensor | torch-spyre | Wraps a PyTorch tensor's metadata plus the `CompositeAddress` from FlexAllocator. The underlying device allocation is freed via `FlexAllocator.deallocate()` when the tensor is no longer referenced. |

#### SpyreStream

The core execution engine in torch-spyre, modeled after CUDA streams. Implements the PyTorch Stream interface (e.g., `torch.spyre.Stream`, `torch.spyre.current_stream()`) and holds a reference to a `RuntimeStream` instance. Responsible for tiling detection, operation construction, and submission to RuntimeStream. Async enqueue, intra-stream FIFO ordering, no cross-stream ordering — all inherited from RuntimeStream.

```cpp
class SpyreStream {
public:
    // Factory: creates a SpyreStream backed by a new RuntimeStream from the given context.
    // mode and priority are forwarded to RuntimeContext::createStream().
    static std::shared_ptr<SpyreStream> create(
        std::shared_ptr<RuntimeContext> context,
        RuntimeStreamMode mode = RuntimeStreamMode::STRICT_ORDERING,
        RuntimeStreamPriority priority = RuntimeStreamPriority::NORMAL);

    // Launch a JobPlan on this stream.
    // tensors: the SpyreTensors corresponding to the kernel's I/O
    //   (order matches SpyreCode tensor list).
    // allow_tiled_launch: if true, Launch handles tiling transparently when shapes
    //   exceed expected_input_shapes. If false, raises SpyreException when shapes
    //   don't match exactly.
    // Asynchronous — returns immediately. Errors during enqueue throw SpyreException.
    // Errors during async execution are deferred (see Error Model).
    void Launch(const JobPlan& plan,
                const std::vector<SpyreTensor>& tensors,
                bool allow_tiled_launch = true);

    // Block until all previously enqueued operations complete.
    // Propagates any deferred errors from async execution as SpyreException.
    void Synchronize();

    // Non-blocking: returns true if all enqueued operations have completed.
    bool Query();

    // Access the underlying RuntimeStream (e.g., for direct RuntimeOperation submission).
    std::shared_ptr<RuntimeStream> getRuntimeStream();

private:
    std::shared_ptr<RuntimeStream> runtime_stream_;
    // Per-stream correction buffer pool (see Correction Buffer Management)
    CorrectionBufferPool correction_buffers_;
};
```

The default SpyreStream is created during device initialization and retrievable via `torch.spyre.current_stream()`. Users can create additional streams via `SpyreStream::create()`.

**Launch(JobPlan, List\<SpyreTensor\>, allow_tiled_launch=true)**

Extracts CompositeAddresses from the SpyreTensors, then compares each tensor's shape against the JobPlan's `expected_input_shapes`. Three cases:

1. **Shapes match exactly** — SpyreStream constructs fully-populated RuntimeOperations from the JobPlan's steps (with all addresses filled in) and submits them to RuntimeStream via `launchOperation()`.

2. **Shapes exceed tile size and `allow_tiled_launch` is true** — SpyreStream infers the tiling dimension(s) and iteration count, then for each iteration constructs RuntimeOperations with updated tensor offsets and submits them to RuntimeStream. See [Tiled Execution](#tiled-execution) for details.

3. **Shapes exceed tile size and `allow_tiled_launch` is false** — SpyreStream raises an exception indicating that the tensor shapes do not match the compiled tile size and tiled launch is not permitted.

**Example 1 — Tensor load (DMA only):**

A JobPlan containing a single `RuntimeOperationH2D` step:

1. **RuntimeOperationH2D** — Copy tensor data from host to device

SpyreStream constructs the operation with the correct addresses and submits it to RuntimeStream.

**Example 2 — Matmul with program correction:**

A JobPlan containing three steps: a `RuntimeOperationHostCallback` (with `ProgramCorrectionStep`), a `RuntimeOperationH2D`, and a `RuntimeOperationCompute`:

1. **RuntimeOperationHostCallback** (correction step) — SpyreStream builds a closure from the `ProgramCorrectionStep` metadata, pre-allocates a host buffer for the correction tensor, and configures the callback to write its output there. Submitted to RuntimeStream, which executes it on the host CPU: converts tensor virtual address metadata using the correction metadata, producing a correction tensor in the pre-allocated buffer.
2. **RuntimeOperationH2D** — SpyreStream configures this with the pre-allocated buffer as the host address and the program allocation's device address. Submitted to RuntimeStream, which transfers the correction tensor to device.
3. **RuntimeOperationCompute** — Submitted to RuntimeStream, which launches the unified binary. Internally, the correction program reads the correction tensor, patches the compute program, then the matmul executes.

All three operations are submitted to RuntimeStream as a vector. RuntimeStream executes them in FIFO order — the program correction runs on the CPU first, then the DMA, then the compute.

`Synchronize()` and `Query()` delegate directly to `RuntimeStream.synchronize()` and `RuntimeStream.query()` respectively. `Synchronize()` is the only blocking method on SpyreStream and surfaces deferred async errors.

#### RuntimeStream

See the RuntimeStream RFC in the internal documentation for full details. Key behaviors relevant to this RFC: RuntimeStream executes `RuntimeOperationHostCallback` steps on the host CPU in stream order, maps DMA and compute operations to control blocks for hardware dispatch, and generates one DMA transfer per chunk internally for interleaved (multi-chunk) allocations.

### Tiled Execution

Tiled execution is handled automatically by `SpyreStream.Launch` when `allow_tiled_launch` is true (the default). When a tensor is larger than the compiled tile size (as specified by the JobPlan's `expected_input_shapes`), SpyreStream reuses the compiled kernel across the full tensor by constructing multiple rounds of RuntimeOperations — one full walk of the JobPlan's steps per iteration, each with updated tensor offsets — and submitting them to RuntimeStream. If `allow_tiled_launch` is false and shapes exceed the tile size, Launch raises an exception.

**Preconditions:**
* Each tensor dimension is greater than or equal to the corresponding compiled tile dimension
* Each tensor dimension is evenly divisible by the corresponding tile dimension
* Tensor strides are consistent with the tiling dimension

**Behavior:**
1. Compare each tensor's shape against the JobPlan's `expected_input_shapes`
2. Infer the tiling dimension(s) and compute `num_iterations = tensor_dim / tile_dim`
3. For each iteration `i`:
   a. Compute per-tensor offset: `offset_i = base_offset + (i * tensor_stride_along_tiled_dim * tile_size * element_size)`
   b. For contiguous (single-chunk) allocations, the offset adjusts within the single chunk — `region_id` stays the same. For interleaved (multi-chunk) allocations, the offset calculation depends on the tiling dimension: tiling along the sharded dimension moves across chunks (different `region_id` values), while tiling along non-sharded dimensions adjusts `offset` within each chunk.
   c. Construct fully-populated RuntimeOperations with updated device addresses (`RuntimeOperationHostCallback` with correction closure and pre-allocated correction buffer, `RuntimeOperationH2D` for correction tensor, `RuntimeOperationCompute`) and submit to RuntimeStream
   d. All operations enqueued asynchronously — sequential within the stream

**Example — Matmul tiling along M (with program correction):**

Kernel compiled for `A[1024, K] * B[K, N] = C[1024, N]`. Actual tensor `A` is `[4096, K]`, `C` is `[4096, N]`:

```
num_iterations = 4096 / 1024 = 4

Iteration 0: A[0:1024, :]    * B → C[0:1024, :]
Iteration 1: A[1024:2048, :] * B → C[1024:2048, :]
Iteration 2: A[2048:3072, :] * B → C[2048:3072, :]
Iteration 3: A[3072:4096, :] * B → C[3072:4096, :]
```

Each iteration, SpyreStream constructs and submits to RuntimeStream:

```
RuntimeOperationHostCallback(correction_closure_i) → RuntimeOperationH2D(correction_tensor_i) → RuntimeOperationCompute(binary)
```

Tensors whose shapes already match the tile (e.g., `B` above) have stride 1 — they are reused across iterations without offset changes.

If any precondition is not met, `Launch()` throws a `SpyreException` with a diagnostic message indicating which tensor, which dimension, and what the mismatch is. No operations are submitted to RuntimeStream.

### Error Model

SpyreStream uses a **sticky error model** (similar to CUDA streams). Once an error occurs on a stream, the stream enters an error state and all subsequent operations on that stream fail immediately without being dispatched.

**Error classification:**

| Error | When | Behavior |
|-------|------|----------|
| Shape mismatch (no tiling) | `Launch`, synchronous | `allow_tiled_launch=false` and shapes don't match → `SpyreException` thrown from `Launch()` |
| Tiling precondition failure | `Launch`, synchronous | Shapes exceed tile but aren't evenly divisible, or strides inconsistent → `SpyreException` from `Launch()` |
| Enqueue failure | `Launch`, synchronous | RuntimeStream rejects operation (e.g., stream destroyed) → `SpyreException` from `Launch()` |
| Hardware error | Async, during execution | RuntimeStream reports error → stream enters error state. Next `Synchronize()` or `Query()` surfaces the error as `SpyreException`. Remaining enqueued operations are not dispatched. |
| Program correction failure | Async, during callback | HostCallback throws → stream enters error state. Same surfacing as hardware error. |

**Recovery:** After an error, the stream is not reusable. The caller must destroy it and create a new one. Device memory allocated by the failed operations is not automatically freed — the caller must deallocate via `FlexAllocator`.

**Tiled execution and errors:** If a tiled launch submits N iterations of operations and iteration K fails, iterations K+1..N are not dispatched (they were enqueued but the sticky error state prevents dispatch). The caller observes the error at the next `Synchronize()`.

### Correction Buffer Management

When a JobPlan contains a step with a `ProgramCorrectionStep`, SpyreStream must provide a host-side buffer for the correction tensor output.

**Allocation strategy:**
- Each SpyreStream maintains a `CorrectionBufferPool` — a simple pool of pinned host memory buffers sized to the maximum correction tensor for the stream's active JobPlans.
- On the first `Launch` of a JobPlan requiring correction, SpyreStream allocates a buffer from the pool (or grows the pool). The buffer size is determined by the JobPlan's correction metadata (the `oshape` from the original SpyreCode `ComputeOnHost` command).
- For tiled execution within a single `Launch()` call, a **single buffer is reused** across iterations. This is safe because operations within a stream are sequential — iteration K's correction is consumed by its H2D before iteration K+1's correction overwrites the buffer.

**Lifecycle:**
- Buffers are owned by the SpyreStream and freed when the stream is destroyed.
- Buffer reuse across `Launch()` calls is permitted (same JobPlan or different, as long as size fits), but requires explicit synchronization.

**Synchronization requirements for buffer reuse:**
- **Within a single `Launch()` call (tiled iterations):** No explicit synchronization needed. FIFO ordering within the stream guarantees that iteration K's H2D completes before iteration K+1's correction callback executes.
- **Across multiple `Launch()` calls on the same stream:** The caller must call `Synchronize()` before the next `Launch()` to ensure the previous H2D has consumed the buffer. Without synchronization, the next `Launch()`'s correction callback may overwrite the buffer while the previous H2D is still reading from it.
- **Future multi-stream support:** When multiple streams are introduced, synchronization will use events and signals (e.g., `stream.waitEvent(event)`) to coordinate buffer reuse across streams without blocking the host.

**Future optimization (not in scope):** Double-buffering for async iteration overlap (unresolved question #4) would require N buffers for N in-flight iterations.

### Front-End Interface

These functions live in `SpyreSDSCKernelRunner` or equivalent torch-spyre component.

```cpp
// Called once after backend compilation. Parses SpyreCode, executes Job Preparation Plan
// (allocations via SpyreAllocator, binary loading via RuntimeOperationH2D on the given stream),
// translates Job Execution Plan into RuntimeOperations with resolved CompositeAddress values,
// and constructs a JobPlan. The resulting JobPlan is cached for reuse across invocations.
// Synchronous — blocks until binary loading completes (calls stream.Synchronize() internally).
std::shared_ptr<JobPlan> PrepareKernel(
    const SpyreCode& spyre_code,
    SpyreStream& stream,
    SpyreAllocator& allocator);

// Called per invocation. Delegates to stream.Launch(plan, tensors, allow_tiled_launch).
// allow_tiled_launch controlled by SPYRE_ALLOW_TILED_LAUNCH env var (default: true).
void LaunchKernel(
    SpyreStream& stream,
    const JobPlan& plan,
    const std::vector<SpyreTensor>& tensors);
```

**PrepareKernel** executes the Job Preparation Plan: `Allocate` → `SpyreAllocator.allocate()` (delegates to `FlexAllocator`, using an `AllocationDirective` with `policy=Bind` targeting the program memory region) → `CompositeAddress`; `InitTransfer` → constructs a `RuntimeOperationH2D` with the binary data and the allocated `CompositeAddress`, submits to `RuntimeStream.launchOperation()`. It then translates the Job Execution Plan into a `JobPlan` with resolved `CompositeAddress` values, cached for reuse.

**LaunchKernel** is the entry point for repeated kernel execution. Currently delegates to `SpyreStream.Launch(job_plan, tensors, allow_tiled_launch)`. The `allow_tiled_launch` value can be controlled by a user environment setting (`SPYRE_ALLOW_TILED_LAUNCH`), allowing users to disable automatic tiling for debugging or to enforce that tensor shapes exactly match the compiled tile size. For interleaved tensors, SpyreStream accounts for the multi-chunk `CompositeAddress` layout when computing tiled offsets and assembling correction metadata. In the future, LaunchKernel may coordinate across multiple streams to interleave execution of independent operation sequences.

### Workflows

#### Workflow 1: Tensor Allocation and Transfer

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  CPUTensor   │────▶│ SpyreAllocator  │────▶│     SpyreStream      │────▶│  RuntimeStream   │
│  (host)      │     │ allocate block  │     │ Launch(JobPlan       │     │ launchOperation  │
│              │     │→CompositeAddress│     │   [H2D])             │     │ (H2D op)         │
│              │     │                 │     │ construct & submit op│     │ → hardware       │
└─────────────┘     └─────────────────┘     └──────────────────────┘     └──────────────────┘
```

1. User creates a `CPUTensor` and calls `.to(device)`
2. SpyreAllocator (via FlexAllocator) allocates a block, producing a `CompositeAddress`
3. torch-spyre creates a JobPlan with a single `RuntimeOperationH2D` step (host_address, composite_address, size)
4. `SpyreStream.Launch(job_plan, tensors)` extracts CompositeAddresses from SpyreTensors, constructs a fully-populated `RuntimeOperationH2D`, and submits it to `RuntimeStream.launchOperation()` — returns immediately
5. RuntimeStream produces a DMA control block and dispatches it to hardware
6. Result is a `SpyreTensor` carrying the device address metadata
7. Host may continue work; data transfer proceeds asynchronously on the stream

#### Workflow 2: Compilation, SpyreCode Translation, and Loading

```
┌──────────┐     ┌───────────┐     ┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│ Inductor │────▶│ Deeptools │────▶│  SpyreCode   │────▶│    torch-spyre       │────▶│  RuntimeStream   │
│ (sdsc)   │     │ (compile) │     │  JSON / SDSC │     │ parse SpyreCode,     │     │ launchOperation  │
└──────────┘     └───────────┘     └──────────────┘     │ run Job Prep Plan,   │     │ (H2D ops)        │
                                                         │ translate Job Exec   │     │ × N              │
                                                         │ Plan → JobPlan       │     │                  │
                                                         │ (torch-spyre internal)│     │                  │
                                                         └──────────────────────┘     └──────────────────┘
```

1. `torch.compile` triggers the inductor frontend, producing SDSC inputs for deeptools
2. Deeptools (backend compiler) produces a `SpyreCode` JSON per SDSC containing a Job Preparation Plan and a Job Execution Plan
3. torch-spyre parses each SpyreCode and executes the **Job Preparation Plan** (run once per SDSC):
   a. `Allocate` → `SpyreAllocator.allocate()` (delegates to `FlexAllocator`, using an `AllocationDirective` with `policy=Bind` targeting the program memory region) → `CompositeAddress`
   b. `InitTransfer` → constructs a `RuntimeOperationH2D` with the binary data and the allocated `CompositeAddress`, submits to `RuntimeStream.launchOperation()`
4. torch-spyre translates the **Job Execution Plan** into a `JobPlan` (torch-spyre internal), resolving the compiler's symbols to `CompositeAddress` values using the allocations from step 3 and storing `expected_input_shapes` from the SpyreCode
5. JobPlans are cached by torch-spyre for reuse across invocations

#### Workflow 3: Detailed Execution — LaunchKernel to Hardware

This diagram shows the full path from `LaunchKernel` through every layer for a matmul with program correction, illustrating how SpyreStream processes a JobPlan and submits the `vector<RuntimeOperation>` to RuntimeStream for hardware dispatch.

```
 torch-spyre                                              flex
┌───────────────────────────────────────────────┐  ┌──────────────────────────────────────┐
│                                               │  │                                      │
│  LaunchKernel(spyre_stream,                   │  │                                      │
│        job_plan, tensors)                     │  │                                      │
│         │                                     │  │                                      │
│         ▼                                     │  │                                      │
│  SpyreStream.Launch(job_plan,                 │  │                                      │
│    tensors, allow_tiled_launch)               │  │                                      │
│         │                                     │  │                                      │
│  Extract CompositeAddresses from SpyreTensors │  │                                      │
│         │                                     │  │                                      │
│  Compare tensor shapes against                │  │                                      │
│  JobPlan.expected_input_shapes:               │  │                                      │
│         │                                     │  │                                      │
│  ┌──────────────────────────────────────┐     │  │                                      │
│  │ shapes match exactly?                │     │  │                                      │
│  │   YES → construct single iteration   │     │  │                                      │
│  │   NO  → allow_tiled_launch?          │     │  │                                      │
│  │         YES → tiled iterations       │     │  │                                      │
│  │         NO  → raise exception        │     │  │                                      │
│  └──────────────────────────────────────┘     │  │                                      │
│         │                                     │  │                                      │
│  (showing exact-match case below;             │  │                                      │
│   see Workflow 4 for tiled case)              │  │                                      │
│         │                                     │  │                                      │
│  Construct fully-populated operations:        │  │                                      │
│    1. RuntimeOperationHostCallback            │  │                                      │
│       (correction closure w/ pre-allocated    │  │                                      │
│        buffer, barrier=true)                  │  │                                      │
│    2. RuntimeOperationH2D                     │  │                                      │
│       (correction buffer → device)             │  │                                      │
│    3. RuntimeOperationCompute                 │  │                                      │
│       (composite_address)                     │  │                                      │
│         │                                     │  │                                      │
│  Submit to RuntimeStream.launchOperation()────┼──┼─▶ RuntimeStream                      │
│                                               │  │   (FIFO — sequential within stream)  │
│                                               │  │                                      │
│                                               │  │   ┌─────────────────────────────┐    │
│                                               │  │   │ 1. HostCallback             │    │
│                                               │  │   │    (correction, on CPU)     │    │
│                                               │  │   │    → correction tensor      │    │
│                                               │  │   ├─────────────────────────────┤    │
│                                               │  │   │ 2. H2D                      │    │
│                                               │  │   │    correction_tensor →      │    │
│                                               │  │   │    program alloc, offset    │    │
│                                               │  │   ├─────────────────────────────┤    │
│                                               │  │   │ 3. Compute                  │    │
│                                               │  │   │    Launch unified binary —  │    │
│                                               │  │   │    correction + matmul      │    │
│                                               │  │   └─────────────────────────────┘    │
│                                               │  │                                      │
│                                               │  │   Dispatches to hardware:             │
│                                               │  │   ┌────────┐ ┌─────────┐             │
│                                               │  │   │  DMA   │→│ Compute │→ done       │
│                                               │  │   │ corr   │ │  launch │             │
│                                               │  │   └────────┘ └─────────┘             │
│                                               │  │                                      │
└───────────────────────────────────────────────┘  └──────────────────────────────────────┘

 Host returns immediately after enqueuing.
 Call SpyreStream.Synchronize() to block until hardware completes.
```

**Step-by-step:**

1. `LaunchKernel(spyre_stream, job_plan, tensors)` is called from torch-spyre
2. Delegates to `SpyreStream.Launch(job_plan, tensors, allow_tiled_launch)`
3. SpyreStream extracts CompositeAddresses from SpyreTensors
4. SpyreStream compares tensor shapes against the JobPlan's `expected_input_shapes`:
   * Shapes match exactly → proceeds with single-iteration construction (this workflow)
   * Shapes exceed tile size and `allow_tiled_launch` is true → tiled iterations (see Workflow 4)
   * Shapes exceed tile size and `allow_tiled_launch` is false → raises exception
5. SpyreStream walks the JobPlan's steps, expanding correction steps into closures:
   * **RuntimeOperationHostCallback**: correction closure built from `ProgramCorrectionStep`, with pre-allocated correction buffer and `barrier=true`
   * **RuntimeOperationH2D**: configured with pre-allocated correction buffer as host address and program allocation device address
   * **RuntimeOperationCompute**: configured with the binary's CompositeAddress
6. SpyreStream submits all operations to `RuntimeStream.launchOperation()` as a vector
7. RuntimeStream executes them in FIFO order: runs program correction on CPU, then dispatches DMA and compute to hardware
8. Host returns immediately; call `SpyreStream.Synchronize()` when results are needed

#### Workflow 4: Tiled Execution

```
 torch-spyre                                              flex
┌───────────────────────────────────────────────┐  ┌──────────────────────────────────────┐
│                                               │  │                                      │
│  LaunchKernel(spyre_stream,                   │  │                                      │
│    job_plan, tensors)                         │  │                                      │
│         │                                     │  │                                      │
│  SpyreStream.Launch(job_plan,                 │  │                                      │
│    tensors, allow_tiled_launch=true)          │  │                                      │
│         │                                     │  │                                      │
│  Extract CompositeAddresses                   │  │                                      │
│  Detect shapes exceed tile size               │  │                                      │
│  allow_tiled_launch=true → proceed            │  │                                      │
│  Infer: 4 iterations (4096/1024)              │  │                                      │
│         │                                     │  │                                      │
│  For each iteration, construct ops            │  │                                      │
│  with updated offsets:                        │  │                                      │
│    ┌─────────┐                                │  │                                      │
│    │ iter 0  │─▶ Callback₀, H2D₀, Compute₀   │  │                                      │
│    ├─────────┤                                │  │                                      │
│    │ iter 1  │─▶ Callback₁, H2D₁, Compute₁   │  │                                      │
│    ├─────────┤                                │  │                                      │
│    │ iter 2  │─▶ Callback₂, H2D₂, Compute₂   │  │                                      │
│    ├─────────┤                                │  │                                      │
│    │ iter 3  │─▶ Callback₃, H2D₃, Compute₃   │  │                                      │
│    └─────────┘                                │  │                                      │
│         │                                     │  │                                      │
│  Submit all 12 ops to RuntimeStream───────────┼──┼─▶ RuntimeStream                      │
│                                               │  │   Executes FIFO:                     │
│                                               │  │   Callback₀→DMA₀→Comp₀→             │
│                                               │  │   Callback₁→DMA₁→Comp₁→             │
│                                               │  │   Callback₂→DMA₂→Comp₂→             │
│                                               │  │   Callback₃→DMA₃→Comp₃              │
│                                               │  │                                      │
└───────────────────────────────────────────────┘  └──────────────────────────────────────┘
```

1. `LaunchKernel(spyre_stream, job_plan, tensors)` is called from torch-spyre
2. Delegates to `SpyreStream.Launch(job_plan, tensors, allow_tiled_launch=true)`
3. SpyreStream extracts CompositeAddresses from SpyreTensors
4. SpyreStream compares tensor shapes against the JobPlan's `expected_input_shapes` — detects shapes exceed tile size
5. `allow_tiled_launch` is true → SpyreStream proceeds with tiled execution
6. SpyreStream infers tiling dimension and `num_iterations = 4096 / 1024 = 4`
7. For each iteration `i`, SpyreStream constructs fully-populated operations with updated addresses:
   * Computes updated tensor offsets for iteration `i`
   * **RuntimeOperationHostCallback**: correction closure built from `ProgramCorrectionStep` with updated metadata for this iteration
   * **RuntimeOperationH2D**: configured with pre-allocated correction buffer and program allocation device address
   * **RuntimeOperationCompute**: configured with the binary's CompositeAddress
8. SpyreStream submits all 12 operations (3 per iteration × 4 iterations) to `RuntimeStream.launchOperation()`
9. RuntimeStream executes all operations in FIFO order: runs program corrections on CPU, dispatches DMA and compute to hardware
10. Host returns immediately; call `SpyreStream.Synchronize()` to block until all iterations complete

#### Workflow 5: End-to-End

```
                                                    # Default SpyreStream created at runtime start

1. tensor_a = torch.randn(4096, 1024)             # Create on CPU
2. spyre_a = tensor_a.to("spyre")                 # JobPlan([H2D]) → SpyreStream → RuntimeStream
3. spyre_b = torch.randn(1024, 1024).to("spyre")  # Same — async, all through default SpyreStream

4. compiled_fn = torch.compile(matmul_fn)          # Lazy — no compilation yet

5. result = compiled_fn(spyre_a, spyre_b)          # First call triggers inductor → deeptools
                                                    # PrepareKernel: SpyreCode → allocate, load binaries → JobPlan (cached)
                                                    # Then: LaunchKernel(job_plan, tensors)
                                                    # Uses default stream (or creates a new one)
                                                    # SpyreStream detects tensor_a [4096,1024] > tile [1024,1024]
                                                    # allow_tiled_launch=true → constructs 4 iterations of ops
                                                    # Submits all ops to RuntimeStream
                                                    # (12 ops: 3 per iteration × 4 iterations)

6. result_cpu = result.to("cpu")                   # JobPlan([D2H]) → SpyreStream → RuntimeStream
```

#### Workflow 6: Multi-Stream (Future Hardware)

```
stream_a = SpyreStream()                           # Stream for layer 1 (holds its own RuntimeStream)
stream_b = SpyreStream()                           # Stream for layer 2 (holds its own RuntimeStream)

# Enqueue independent work on separate streams
LaunchKernel(stream_a, job_plan_1, tensors_1)      # Async — SpyreStream submits to its RuntimeStream
LaunchKernel(stream_b, job_plan_2, tensors_2)      # Async — no ordering w.r.t. stream_a

# With current hardware: RuntimeInterface serializes both streams' operations
# With future hardware: Both streams may execute concurrently

stream_a.Synchronize()
stream_b.Synchronize()
```

## **Metrics**

TBD

## **Drawbacks**

TBD

## **Alternatives**

TBD

## **Prior Art**

### CUDA Streams and SpyreStream

The SpyreStream + RuntimeStream architecture is modeled after CUDA streams and shares the two fundamental guarantees: **async enqueue** (all methods return control to the host immediately) and **intra-stream FIFO ordering** (operations within a stream execute sequentially, with no ordering guarantees across streams).

The key difference is where work submission lives. A CUDA stream is a **passive handle** — you pass it as a parameter to external API calls like `cudaMemcpyAsync(dst, src, size, kind, stream)` or kernel launches via `<<<grid, block, sharedMem, stream>>>`. SpyreStream is an **active object** — operations are submitted as methods on the stream itself (`stream.Launch(job_plan, ...)`, `stream.Synchronize()`). SpyreStream handles orchestration (tiling, operation construction) and delegates hardware dispatch to RuntimeStream via `launchOperation()`.

The following table maps CUDA stream capabilities to their SpyreStream/RuntimeStream equivalents:

| Capability | CUDA Stream | SpyreStream / RuntimeStream |
|------------|-------------|------------|
| Copy | `cudaMemcpyAsync(dst, src, size, kind, stream)` | `RuntimeOperationH2D`/`D2H` steps in a JobPlan, submitted via `RuntimeStream.launchOperation()` |
| Launch | `kernel<<<grid, block, sharedMem, stream>>>()` | `Launch(JobPlan, tensors)` — no grid/block dims (not SIMT); auto-tiling when shapes exceed tile size |
| Tiled launch | User changes grid dims | Automatic within `Launch` — iterates over tiles with updated offsets |
| Synchronize | `cudaStreamSynchronize(stream)` | `Synchronize()` — identical semantics |
| Events | `cudaEventRecord` / `cudaStreamWaitEvent` | Not present (see unresolved question #1) |
| Query | `cudaStreamQuery()` | `Query()` — identical semantics |
| Priorities | `cudaStreamCreateWithPriority()` | Supported via `RuntimeStreamPriority` |
| Host callbacks | `cudaLaunchHostFunc()` | `RuntimeOperationHostCallback` executed on host CPU in stream order |
| Default stream | NULL stream with legacy blocking semantics | Not yet specified |

SpyreStream exposes 3 methods (Launch, Synchronize, Query) vs. CUDA's dozens. The omitted features are captured as unresolved questions.

### SpyreStream and RuntimeStream — Compound Operation Sequences

In CUDA, a kernel launch is self-contained. Spyre operations have inherent **compound structure** — a single matmul requires HostCallback (correction) → H2D → Compute (3 operations). The closest CUDA analogs are vendor libraries like cuBLAS/cuDNN, which internally issue multi-step sequences — but in the Spyre stack, compound decomposition is built into SpyreStream because every Spyre compute requires it.

Similarly, CUDA kernels accept arbitrary dimensions via `<<<grid, block>>>`, so tiling is the user's responsibility. Spyre kernels are compiled for fixed tile shapes, so SpyreStream handles tiling automatically within `Launch`.

## **How we teach this**

TBD

## **Unresolved questions**

1. **Stream events / synchronization primitives**: SpyreStream will need inter-stream synchronization (e.g., CUDA-style events) to express dependencies between streams without full synchronization. The requirement is clear, but the implementation approach is still open.

2. **Reduction tiling**: When tiling along the reduction dimension (K in matmul), partial results must be accumulated. Should the runtime handle accumulation internally, or should this be expressed as a separate step in the operation sequence?

3. **Multi-dimensional tiling**: If both M and N exceed the tile size, the runtime needs a nested loop. Should tiled execution support arbitrary nesting, or should this be constrained to single-dimension tiling with multi-dim handled by the compiler?

4. **Async iteration overlap**: Currently each tiled iteration within a stream is sequential. Could we use separate streams or double-buffering to overlap iteration N+1's data movement with iteration N's compute?

5. **Program correction across PF/VF modes**: The program correction closure (built from `ProgramCorrectionStep`) takes LogicalAddresses as input — these are extracted from the `CompositeAddress` chunks provided directly by torch-spyre. Since LogicalAddress is a single type (`region_id` + `offset`) in both modes, the correction closure can process them uniformly — but the correction program on device may still need to interpret `region_id` differently (firmware lookup index vs. physical address). How should this mode distinction be communicated to the correction program?

6. **Pipelining across operation sequences**: The hardware supports a 3-stage pipeline across operations: while one `RuntimeOperationCompute` executes, the correction `RuntimeOperationHostCallback` and `RuntimeOperationH2D` for the *next* operation can overlap (since DMA does not engage AI cores). This inter-sequence pipelining is distinct from unresolved question #4 (which covers intra-sequence tiled iteration overlap). How should SpyreStream and RuntimeStream coordinate to exploit this overlap — should SpyreStream submit operations from multiple sequences to RuntimeStream concurrently, or should RuntimeStream handle pipelining internally?

7. **Symbolic shapes as program correction inputs**: Symbolic shapes (not just addresses) can be inputs to the program correction closure if the kernel was compiled with symbolic shapes. Should the frontend (inductor) use program selection (multiple static kernels) or pass symbolic shape values through to the runtime for the correction closure to resolve? This affects what the correction closure receives as input and whether the runtime needs to handle dynamic shapes.

8. **Correction metadata for interleaved tensors**: When a tensor has a multi-chunk `CompositeAddress` (interleaved across memory domains), the program correction closure receives multiple `LogicalAddress` values for that tensor — one per chunk. The correction metadata must encode this multi-chunk layout so that the correction program can address data across multiple segments. How should multi-chunk layouts be represented in the correction metadata? Should the correction program be aware of the number of chunks and their segment locations, or should SpyreStream flatten the multi-chunk layout into a format the existing correction program can consume?

9. **Binary delivery mechanism**: SpyreCode currently references binaries via a file path on disk. The performance and security implications of the underlying filesystem (tmpfs, NFS, S3, etc.) should be evaluated. Future revisions may support in-memory binary delivery to avoid file I/O overhead and filesystem-dependent behavior.

10. **Host-side pinned memory for DMA**: Host-to-device and device-to-host DMA transfers require host-side pinned (page-locked) memory. Should the runtime use bounce buffers (runtime pins a fixed region, copies data through it), or should PyTorch allocate directly into pinned regions to avoid the extra copy? This affects DMA transfer performance and should be coordinated with the SpyreAllocator RFC.

## Resolution

TBD

### Level of Support

TBD

#### Additional Context

TBD

### Next Steps

TBD

#### Tracking issue

TBD

#### Exceptions

TBD
