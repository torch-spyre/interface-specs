# Overview

The SuperDSC-Bundle API defines the interface between [torch-spyre](https://torch-spyre.readthedocs.io/) frontend compiler to PyTorch and the Spyre backend compiler ([Deeptools](https://research.ibm.com/publications/deeptools-compiler-and-execution-runtime-extensions-for-rapid-ai-accelerator)) that generates the code to run on the [IBM Spyre AI Accelerator](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator). 

SuperDSC stands for ***Super Design Space Configuration*** and is a JSON-based IR designed to describe the tile-level compute graph for the 32 cores of Spyre.
A SuperDSC-Bundle enables the expression of data-parallel mappings for complex kernels across multi-core accelerators.
Follow the link for a description of the [SuperDSC-Bundle Interface Specification](../SuperDSC-Bundle.md).

The figure below illustrates what is called the Spyre Stack where a user-written PyTorch program is compiled by torch-spyre to generate a set of files known as the **SuperDSC-Bundle**. A PyTorch file may translate into several SuperDSC-Bundles, each one being composed of a ***bundle.mlir*** file and several ***sdsc_\*.json*** files, each JSON file describing a torch operation (a full list of supported PyTorch operations can be found in [torch-spyre](https://torch-spyre.readthedocs.io/)).
Each SuperDSC-Bundle is compiled by **DeepTools** to generate the assembly code that runs on the Spyre AI Accelerator Card. Furthermore, the compiler also generates the execution plan that manages the execution of an operation as well as intra-memory data movement on the Spyre Card.

![High-Level View of SuperDSC-Bundle API within torch-spyre stack](../spec/json/figures/torch_spyre_backend_flow.png)

*Figure 1: High-Level View of SuperDSC-Bundle API within torch-spyre stack*

SuperDSC is a self-contained compiled artifact that describes everything the Spyre hardware needs to execute a single scheduled operation deterministically. The top-level structure contains core fold properties, work-slice mappings, and a per-core execution schedule. A `dscs_` array holds one or more DesignSpaceConfig entries. Each entry is a complete description of one compute configuration, and contains the following elements.

- **Core fold properties** (`coreFoldProp_`, `numWkSlicesPerDim_`, `coreIdToWkSlice_`): how to divide the iteration space across cores. Any operation can use up to 32 cores — the current core count on Spyre, though this limit may increase in future hardware generations. For a tensor of shape (1024, 256), this encodes how many rows each core processes. The encoding gives each core an equal number of sticks and keeps each core within its addressable device memory limit.
- **Tensor descriptors** (`labeledDs_`, `primaryDsInfo_`): for each tensor argument, the tiling structure defines which dimensions are stick dimensions, how the host-side shape maps to device-side tiles, memory residency (HBM vs. LX scratchpad), data format, and which dimensions each tensor iterates over fully vs. which are summed over (contracted) as in the K dimension of a matmul.
- **Schedule tree** (`scheduleTree_`): a list of allocate nodes (one per tensor) that specify memory placement (HBM or LX scratchpad), dimension ordering, per-core start addresses via fold mappings, and coordinate information encoding how each dimension is split across cores with affine transformations.
- **Data staging** (`dataStageParam_`): per-core dimension sizes for steady-state and epilogue passes, describing how data is partitioned for transfer into scratchpad.
- **Compute operations** (`computeOp_`): one entry per operation, encoding the execution unit (PT or SFP), operation name, data format, fidelity, and the input/output tensor references from labeledDs_.

Folding is a central concept in SuperDSC. A single parameterized artifact can represent multiple execution variants across time steps and cores without recompilation. Fold properties use affine transformations (alpha * index + beta) to compute per-core coordinates and addresses. One JSON file describes the behavior of all active cores compactly instead of duplicating the description for each core.


## Hardware Abstraction Model

SuperDSC-Bundle views Spyre hardware at the **data-parallel level of hardware abstraction**:

- **Multiple cores**, each with:
  - Compute engine (AIU)
  - Scratchpad memory (LX)
- **On-chip interconnect fabric** connecting cores and off-chip memory banks
- **Off-chip memory** (HBM/DDR) for large data storage

![Hardware abstraction of multi-core accelerator embodied in SuperDSC-Bundle](../spec/json/figures/data_parallel_hw_abstraction.png)

*Figure 2: Hardware abstraction of multi-core accelerator embodied in **SuperDSC-Bundle***

This abstraction enables frontend compilers to express:

- Complex kernels comprised of operation sequences
- Work division (computation split) across Spyre cores
- Tensor placement in DDR memory or LX scratchpad
- Static or symbolic shapes and start addresses

The SuperDSC-Bundle output is consumed by Deeptools to produce `SpyreCode`. For a full description of `SpyreCode` and its symbolic correction mechanism, see [Execution Model](../SuperDSC-Bundle.md#execution-model).

## API Components

The API consists of two primary components:

1. **MLIR Bundle File** (`.mlir`) — Orchestrates execution flow and symbol management
2. **SDSC JSON Files** (`.json`) — Defines individual operations and their core mappings

All `sdsc_*.json` files MUST conform to the [SDSC Bundle JSON Schema](../spec/sdscbundle-schema.json).
The schema provides machine-readable type constraints, required-field enforcement, and enum
validation for every object in the hierarchy. It is the normative reference for structural
correctness.

## Important Notes

For supported operation categories (including data-shuffle operations such as stick-breaking, gather, and scatter) and tensor allocation rules, see the [API Components note](../SuperDSC-Bundle.md#api-components) in the spec, [ComputeOperation](../spec/json/computeoperation.md), and [Stick Layout Constraints](../spec/json/stick-layout-constraints.md).

---

## Examples

### Simple MLIR Examples

Minimal `.mlir` files illustrating the core bundle patterns in isolation.

| File | Description |
|---|---|
| [`1-single-no-sym.mlir`](1-single-no-sym.mlir) | Single SDSC with no symbols — all start addresses, work division, and sizes are encoded directly in the JSON file. The bundle only invokes `sdsc_execute` with no arguments. |
| [`2-softmax-no-sym.mlir`](2-softmax-no-sym.mlir) | Series of six SDSCs without symbols, implementing softmax (max → sub → exp → sum → reciprocal → mul). Shapes and addresses are fixed in each JSON file. |
| [`3-single-fake-sym.mlir`](3-single-fake-sym.mlir) | Single SDSC where four constant start addresses are passed as symbols (`symbol_ids=[-1, -2, -3, -4]`). Illustrates symbolic address passing even when values are compile-time constants. |
| [`4-loop-multi.mlir`](4-loop-multi.mlir) | Two SDSCs inside a `scf.for` loop. Per-core start addresses are computed each iteration using `affine.apply` maps, with per-core tensor splits driving separate symbol values per core. |

### MLIR Examples

[`MLIR-examples.md`](MLIR-examples.md) — Annotated usage examples covering the common bundle patterns:

- Single operation with no dynamic symbols
- Multi-operation sequence (e.g. softmax decomposition)
- Symbolic start addresses passed from the bundle to an SDSC
- Tiled loop with affine address arithmetic

### MLIR Complete Example

[`MLIR-complete-example.md`](MLIR-complete-example.md) — End-to-end walkthrough of a softmax bundle (`softmax.mlir`) with dynamic `input_arg` tensor parameters, per-core affine address computation, and fully annotated symbolic bindings.

### JSON Examples

[`JSON-examples.md`](JSON-examples.md) — Complete JSON examples for SDSC files, including a fully worked simple GELU operation showing all required and optional fields across `SuperDsc`, `DesignSpaceConfig`, `scheduleTree_`, `labeledDs_`, and `computeOp_`.

---

[↑ Spec Map](../SuperDSC-Bundle.md#spec-map)|
