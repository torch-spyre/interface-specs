# Overview

The SuperDSC-Bundle API defines the interface between [torch-spyre](https://torch-spyre.readthedocs.io/) frontend compiler to PyTorch and the Spyre backend compiler ([Deeptools](https://research.ibm.com/publications/deeptools-compiler-and-execution-runtime-extensions-for-rapid-ai-accelerator)) that generates the code to run on the [IBM Spyre AI Accelerator](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator). 

SuperDSC stand for ***Super Design Space Configuration*** and is a JSON based IR designed to describe the tile-level compute graph for the 32 cores of Spyre.
A SuperDSC-Bundle enables the expression of data-parallel mappings for complex kernels across multi-core accelerators.
Follow the link for a description of the [SuperDSC-Bundle Interface Specification](https://github.com/jneves60/interface-specs/blob/main/0248-SdscBundleSpec/SuperDSC-Bundle.md).

The Figure below illustrates what is called the Spyre Stack where a user written PyTorch program is compiled by torch-spyre to generate a sets of files known as the **SuperDSC-Bundle**. A PyTorch file may translate into several SuperDSC-Bundles, each one being composed of a ***bundle.mir*** file and several ***sdsc_\*.json*** files, each json file describing a torch operation (A full list of supported operations can be found in [torch-spyre](https://torch-spyre.readthedocs.io/)). 
Each SuperDSC-Bundle is compiled by **DeepTools** to generate the assembly code that runs on the Spyre AI Accelerator Card. Furthermore, the compiler also generates the execution plan that manages the execution of an operation as well as intra-memory data movement on the Spyre Card.

![High-Level View of SuperDSC-Bundle API within torch-spyre stack](figures/torch_spyre_backend_flow.png)

*Figure 1: High-Level View of SuperDSC-Bundle API within torch-spyre stack*

SuperDSC is a self-contained compiled artifact that describes everything the Spyre hardware needs to execute a single scheduled operation deterministically. The top-level structure contains core fold properties, work-slice mappings, and a per-core execution schedule. A `dscs_` array holds one or more DesignSpaceConfig entries. Each entry is a complete description of one compute configuration, and contains the following elements.

- **Core fold properties** (`coreFoldProp_`, `numWkSlicesPerDim_`, `coreIdToWkSlice_`): how to divide the iteration space across 32 cores. For a tensor of shape (1024, 256), this encodes how many rows each core processes. The encoding gives each core an equal number of sticks and keeps each core within its addressable device memory limit.
- **Tensor descriptors** (`labeledDs_`, `primaryDsInfo_`): for each tensor argument, the tiling structure define which dimensions are stick dimensions, how the host-side shape maps to device-side tiles, memory residency (HBM vs. LX scratchpad), data format, and which dimensions each tensor iterates over fully vs. which are summed over (contracted) as in the K dimension of a matmul.
- **Schedule tree** (`scheduleTree_`): a list of allocate nodes (one per tensor) that specify memory placement (HBM or LX scratchpad), dimension ordering, per-core start addresses via fold mappings, and coordinate information encoding how each dimension is split across cores with affine transformations.
- **Data staging** (`dataStageParam_`): per-core dimension sizes for steady-state and epilogue passes, describing how data is partitioned for transfer into scratchpad.
- **Compute operations** (`computeOp_`): one entry per operation, encoding the execution unit (PT or SFP), operation name, data format, fidelity, and the input/output tensor references from labeledDs_.

Folding is a central concept in SuperDSC. A single parameterized artifact can represent multiple execution variants across time steps and cores without recompilation. Fold properties use affine transformations (alpha * index + beta) to compute per-core coordinates and addresses. One JSON file describes the behavior of all 32 cores compactly instead of duplicating the description for each core.


## Hardware Abstraction Model

SuperDSC-Bundle views Spyre hardware at the **data-parallel level of hardware abstraction**:

- **Multiple cores**, each with:
  - Compute engine (AIU)
  - Scratchpad memory (LX)
- **On-chip interconnect fabric** connecting cores and off-chip memory banks
- **Off-chip memory** (HBM/DDR) for large data storage

![Hardware abstraction of multi-core accelerator embodied in SuperDSC-Bundle](figures/data_parallel_hw_abstraction.png)

*Figure 2: Hardware abstraction of multi-core accelerator embodied in **SuperDSC-Bundle***

This abstraction enables frontend compilers to express:

- Complex kernels comprised of operation sequences
- Work division (computation split) across RaPiD cores
- Tensor placement in DDR memory or LX scratchpad
- Static or symbolic shapes and start addresses

The SuperDSC-Bundle output is used by the Deeptools backend compiler to produce the `SpyreCode`
consumed by the Spyre Card. The `SpyreCode` contains the job binary, a job plan and other
compiled artifacts. In the case where either the start-address and/or shapes are symbolic,
`SpyreCode` allows for the program binary to contain variables that need to be substituted
or corrected before execution. The mechanism to perform program correction just-in-time before
the job is launched onto spyre is also produced by the backend compiler as part of `SpyreCode`.

## API Components

The API consists of two primary components:

1. **MLIR Bundle File** (`.mlir`) — Orchestrates execution flow and symbol management
2. **SDSC JSON Files** (`.json`) — Defines individual operations and their core mappings

## Important Notes

### Data Shuffle Operations Support

DesignSpaceConfig can represent **BOTH** deep learning operators AND data-shuffle operations:

- **Deep learning operators**: Matmul, convolution, activations, reductions, etc.
- **Data-shuffle operations**: Stick-breaking, non-stick breaking, gather, scatter

### Tensor Allocation vs Work Division

**Critical:** Tensor allocation need NOT be compatible with compute work division. This means:

- Data in one core can be directly available for compute in another core
- The backend compiler will ensure proper data movement across cores
- **Current Limitation:** This functionality is not yet available in the backend; it will be implemented in a future iteration

### Padding for Chunked Dimensions

When a padded dimension is chunked across cores:

- Set front/back padding to `-1` in the "core" data stage
- This signals special handling for distributed padding

### Stick Constraint Ripple Effects

Stick constraints can cause ripple effects across tensors:

- A tensor may need padding in its non-stick dimension
- This occurs when that dimension appears in the stick of another tensor feeding the same operation
- Required to ensure dimension span consistency across all tensors

### Input2 Padding in Matmul

In BatchMatmul operations:

- Input2 must be padded along `reduction_dim` (even though it's not in Input2's stick)
- Required because `reduction_dim` is part of Input1's stick
- Ensures dimension span consistency between inputs

---

| [↑ Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: Bundle API →](MLIR-bundle-API.md) |
|:--|:--:|--:|
