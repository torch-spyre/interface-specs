# **SuperDSC-Bundle Interface Specification**

**Authors:**
* @lupalby
* @Prasanth-Chatarasi
* @bmahjour
* @viji560
* @vswagath1989

## **Summary**

This RFC describes the `SuperDSC-Bundle`, the interface between torch-spyre frontend compiler and the Spyre backend compiler (Deeptools).

## **Motivation**
The interface is essential to connect the torch-spyre frontend compiler with the Deeptools backend compiler to successfully map any operation to Spyre.

## **Proposed Implementation**

`SuperDSC-Bundle` views the sypre hardware at the data-parallel level of hardware abstraction. In this abstraction, Sypre is viewed as having multiple cores, with each core having a compute engine and a scratchpad memory. The cores are interfaced with each other and off-chip memory banks using an on-chip interconnect fabric.

<p align="center">
  <img src="data_parallel_hw_abstraction.png" alt="data_parallel_hw_abstraction" width="450"/>
</p>
<p align="center">
  Figure 1. Hardware abstraction of multi-core accelerator embodied in `SuperDSC-Bundle`
</p>  

`SuperDSC-Bundle` enables the frontend compilers to express data-parallel mappings of:
* Complex kernels comprised of a sequence of operations
* The work division (or computation split) across RaPiD cores of Spyre for each operation,
* The placement of input/output tensors to each operation either in DDR memory or LX scratchpad of the RaPiD cores
* Shapes of the operations and tensors is allowed to be static or symbolic
* The start address of the tensors in DDR and LX is allowed be a fixed number or symbolic

The `SuperDSC-Bundle` specification is used by the Deeptools backend compiler to produce `SpyreCode` containing the job binary, a job plan and other compiled artifacts. In the scenario, where either the start-address and/or shapes are symbolic, `SpyreCode` allows for the program binary to contain variables that need to be substituted or corrected before execution. The mechanism to effect program correction just-in-time before the job is launched onto spyre is also produced by the backend compiler as part of `SpyreCode`.

NOTES:
* Frontend/Backend compiler interface will transition to a new interface called Kernel Tile Intermediate Representation (KTIR) in the future (https://github.com/torch-spyre/torch-spyre/blob/main/RFCs/0682-KtirSpec/0682-KtirSpecRFC.md)
* `SuperDSC` (without bundle capability) is the current interface between Deeptools frontend compiler and Deeptools backend compiler
* `SpyreCode` is tracked through: https://github.com/torch-spyre/torch-spyre/issues/277

## Structure of SuperDSC-Bundle

The backend expects the frontend to produce multiple output files, that work in conjuction to instruct the backend on how one or multiple programs can be compiled and executed in a sequence as part of a complex kernel. The expectation is to receive:
* one or more sdsc.json files, each describing a Spyre operation
* one mlir file with the SuperDsc-Bundle IR

### SuperDSC-Bundle intermediate representation in mlir

This intermediate representation (IR) in mlir conveys a complex kernel made of one or multiple operations. This IR can be used to chain together multiple operations in a sequence and/or to add loops around them. This is achieved through new and existing mlir operations.

#### `sdscbundle.sdsc_execute` op

The central operation in this IR is a new operation we introduce to instantiate a SuperDSC in the execution plan of the complex kernel. This is an example:

```mlir
sdscbundle.sdsc_execute (%A_start_address, %B_start_address) {sdsc_filename="sdscA.json", symbol_ids=[-1, -2]}
```

The operation does not **return** anything.

The operation **attributes** are:
* `sdsc_filename` the relavite path and filename of the specific sdsc.json to instantiate, relative to the location of the mlir file. As one bundle can contain multiple sdsc, different names must be chosen for each json file and here we can refer to the exact one of interest.
* `symbol_ids` list of the symbol ids used inside the sdsc, if any, to represent symbolic start addresses or sizes

The operation **operands** are the SSA variables corresponding to the values to be assigned to the symbols listed in `symbol_ids`, passed in the same order as the symbol ids. These values can be constants (`arith.constant`), or the result of affine expressions, like [`affine.apply`](https://mlir.llvm.org/docs/Dialects/Affine/#affineapply-affineaffineapplyop). The affine expressions can be comprised of constants and loop iterators. Having actual symbols in mlir will be supported through the next revision of the spec.

The `symbols_ids` must be unique in the bundle i.e., symbols ids cannot be recycled across sdscs that are part of the same bundle (unless they take the same value).

#### `sdscbundle.device_mem_allocate` op

A bundle often needs device memory that is not one of the kernel's inputs or outputs: buffers holding intermediate tensors handed from one SDSC to the next, and scratch space used within an SDSC. `sdscbundle.device_mem_allocate` lets the frontend request such memory from the backend, instead of keeping its address symbolic and having it supplied to the bundle from outside. The requested memory is held for the full `SuperDSC-Bundle`: it is reserved before the first SDSC in the bundle executes and is not deallocated at any intermediate point within the bundle.

```mlir
%pool = sdscbundle.device_mem_allocate 65536 bytes : index
```

The operation **returns** the byte address of the first byte of the allocated buffer, as an `index`. This is a device address in the same address space as the start addresses used inside `sdsc.json`, so it can be passed to `sdscbundle.sdsc_execute` as the value of a symbolic start address.

The operation **attributes** are:
* `size` the size of the requested buffer in bytes. It must be a positive constant; symbolic sizes are not supported. A request can be for up to ~15GB: the memory is carved out of a single memory segment whose maximum size is 16GB, of which 1GB is reserved for backend-generated programs and correction-related tensors.

The operation takes no **operands**.

The requested memory is a single contiguous range, and its contents are undefined at allocation.

##### Lifetime

The allocation lives for the entirety of the kernel. The backend reserves the requested bytes for the whole job, so the buffer is valid from the start of the bundle through the end of its last SDSC, and is released only once the kernel completes. There is no matching deallocate operation, and the backend does not reuse the range for anything else within the kernel. Consequently:
* the operation should appear in the entry block of the bundle function, outside of any `scf.for`. An allocation written inside a loop still reserves one single buffer, not one buffer per iteration.
* each `device_mem_allocate` in the bundle gets its own non-overlapping range, so the device memory a bundle requires is the sum of all of its requests, and it is that sum which must stay within the ~15GB budget above. A frontend that wants to reuse space across tensors with non-overlapping live ranges should request a single pool and sub-allocate it itself, as described below.

##### Sub-allocation

Individual buffer addresses are derived from the returned base address by adding a constant offset (`arith.addi`), and the result is passed to `sdscbundle.sdsc_execute` in place of an absolute start address. When a pool is carved up this way the frontend owns its layout: it must ensure that buffers whose live ranges overlap are given disjoint offset ranges, and that each offset satisfies the alignment required by the tensor placed there.

This example allocates a 64KB pool and splits it into four 16KB buffers, reusing the first two as inputs of a later SDSC:

```mlir
%pool = sdscbundle.device_mem_allocate 65536 bytes : index

%off_0     = arith.constant 0 : index
%off_16384 = arith.constant 16384 : index
%off_32768 = arith.constant 32768 : index
%off_49152 = arith.constant 49152 : index

%addr_0     = arith.addi %pool, %off_0     : index   // sdsc_0 output, sdsc_2 input
%addr_16384 = arith.addi %pool, %off_16384 : index   // sdsc_1 output, sdsc_2 input
%addr_32768 = arith.addi %pool, %off_32768 : index   // sdsc_2 output, sdsc_3 input
%addr_49152 = arith.addi %pool, %off_49152 : index   // sdsc_3 scratch

sdscbundle.sdsc_execute (%arg_0, %addr_0) {sdsc_filename="sdsc_0.json", symbol_ids=[-1, -2]}
sdscbundle.sdsc_execute (%arg_1, %addr_16384) {sdsc_filename="sdsc_1.json", symbol_ids=[-3, -4]}
sdscbundle.sdsc_execute (%addr_0, %addr_16384, %addr_32768) {sdsc_filename="sdsc_2.json", symbol_ids=[-5, -6, -7]}
sdscbundle.sdsc_execute (%addr_32768, %addr_49152, %arg_2) {sdsc_filename="sdsc_3.json", symbol_ids=[-8, -9, -10]}
```

#### Loops
Loops are represented using [`scf.for`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop) operation borrowed from MLIR's SCF dialect. This allows SuperDSC-Bundle to describe complex kernels with multiple levels of loops and multiple SDSCs.

We do not support loop carried variables, the only supported scenario is the direct use of the induction variable (loop iterator).

The loop bound should be a constant. Symbolic loop bound will be enhanced in the next revision of the spec.

### `sdsc.json` filling

The individual fields of the SuperDSC to express an operation and its core mapping is described below:
* Each SuperDSC contains a vector to express core work mapping for an operation.
  * Field: `sdsc.dscs_`
  * With balanced work division, only one entry in the vector is needed
    * `sdsc.dscs_[0]`
  * **DesignSpaceConfig can represented BOTH deep learning operators and data-shuffle operations** (stick-breaking, non-stick breaking, gather, scatter)
* Operation(s) to perform: `sdsc.dscs_[0].computeOp_`
  * High level operation selection, like GELU, or BATCHMATMUL
    * Field `OpFuncs opFuncName` in `sdsc.dscs_[0].computeOp_[0]`
  * Set the format to execute the operation in (DL16, FP32, …)
    * Field `DataFormats dataFormat_` in `sdsc.dscs_[0].computeOp_[0].attributes_`
  * List input/output tensors involved with the op
    * Fields `std::vector<LabeledDsInfo*> inputLabeledDs`, `std::vector<LabeledDsInfo*> outputLabeledDs` and `std::vector<LabeledDsInfo*> indirectAccessIndexLabeledDs`
* Work division across cores
  * cores involved
    * int `numCoresUsed_` in `sdsc`
    * int `numCoresUsed_` in `sdsc.dscs_[0]`
    * `std::vector<int> coreIdsUsed_` in `sdsc.dscs_[0]`
    * `std::unique_ptr<FoldDimProp>` `coreFoldProp_`, `coreletFoldProp_` in `sdsc`
      * for core fold, factor=maxCoreId
      * for corelet fold, factor=2
      * use these FoldDimProps when initializing any FoldManager below
  * work division
    * number of slices per dimension
      * `std::map<PrimaryDimTypes, int> numWkSlicesPerDim_` in sdsc
    * core to slice mapping per dimension
      * `std::map<int, std::map<PrimaryDimTypes, int>> coreIdToWkSlice_` in sdsc
  * fill data stage parameters
    * total sizes per dimension (across all cores)
      * `DataStructDims N_` in `sdsc.dscs_[0]`
    * sizes per dimension for a single core
      * `std::map<int, dsc2::DataStage> dataStageParam_` in `sdsc.dscs_[0]`
    * add one entry with key 0, and fill `ss_` and `el_` with same data (name should be “core”)
    * for window/padded operations, add padding information in both datastages above
      * `std::map<PrimaryDimTypes, DimPaddingSizes> paddingSizes_` in `DataStructDims`
      * capture information about front/back padding, stride, related kernel dimension
      * if a padded dimension is chunked across cores, set front/back padding to -1 in “core” datastage
* Input and Output tensors
  * add one entry in `std::vector<LabeledDsInfo> labeledDs_` in `sdsc.dscs_[0]`
  * add one AllocateNode in `sdsc.dscs_[0].scheduleTree_`
  * set data format (dl16, fp32, etc)
    * DataFormats dataFormat_ in `sdsc.dscs_[0].labeledDs_[x]`
  * memory residency (HBM vs LX)
    * `SenComponents component_` in AllocateNode
  * start address per core
    * `FoldManager<int64_t> startAddressCoreCorelet_` in AllocateNode
    * first fold is for cores, set as Map fold type
    * second fold is for corelets, set as Const fold type
    * When the start address is symbolic, set `isStartAddrSymbolic_` boolean to True: 
  * layout
    * stick layout/sizes
      * add entry in `std::map<DsTypes, PrimaryDsInfo> primaryDsInfo_` in `sdsc.dscs_[0]`
      * fill `std::vector stickDimOrder_` and `std::vector stickSize_` in `primaryDsInfo_` entry
      * multiple tensors can share same `primaryDsInfo_` entry if they have same stick layout
    * Layout outside the stick
      * fill `std::vector<PrimaryDimTypes> layoutDimOrder_` in `primaryDsInfo` and `AllocateNode`
      * fill `std::vector<int> maxDimSizes_` in `AllocateNode`
        * set all to -1 (unbound) or to the page size in case of paged value tensor
        * order matches `layoutDimOrder_` in `AllocateNode`
    * back-gaps
      * fill `std::map<PrimaryDimTypes, std::map<int, int>> backGapCore_` in `AllocateNode` with the gaps in number of elements
      * primary key is the dimension in which to apply the gap
      * secondary `int` key is the core id
        * useful when the gap is present in an LX allocation
        * for HBM allocations, set to -1
      * only back-gaps are considered, as front gaps should be handled by simply moving forward the start address
    * scale per dim (to represent reduction/broadcast)
      * 1 is normal, -1 is reduced/broadcasted, -2 is reduced/broadcasted stick dimension
      * `std::vector<double> scale_` in `sdsc.dscs_[0].labeledDs_[x]`
      * order matches layoutDimOrder_ in primaryDsInfo
    * For indirectly accessed tensors (e.g. Paged tensors)
      * Fill maxDimSizes_ in AllocateNode of value tensor to set page size
      * Mark value/index allocations as such and link them to each other
        * enum class `IndirectAllocType indirectAllocType_` in AllocateNode
        * `AllocateNode- relatedIndirectAccessAlloc_` in AllocateNode
  * Tensor coordinates per dimension
    * `CoordinateType<CoordinateBaseType> allocateCoordinates_` in AllocateNode
    * coordinates arrangement is expressed through a sequence of nested simple affine expressions (alpha*index + beta)
      * “factor” is the cardinality of the fold
    * There is no limit on the number of element arrangement folds
    * The combined factor (multiplied) should correspond to the number of elements in that dimension for the tensor
    * Example:
      * coordinate sequence: 0, 1, 2, 3, 64, 65, 66, 67, 4, 5, 6, 7, 68, 69, 70, 71
      * coordinates arrangement (outer to inner)
        * alpha=4, beta=0, factor=2
        * alpha=64, beta=0, factor=2
        * alpha=1, beta=0, factor=4
      * coordinates also require spatial folds
        * core fold
          * for HBM, N/A → alpha=1, factor=1
          * for LX, alpha=coordinate offset across slices, factor=number of slices in dimension
        * corelet fold: N/A → alpha=1, factor=1
        * row fold: N/A → alpha=1, factor=1
      * **NOTE**: the tensor allocation need NOT be compatible with compute work division. For supported static affine LX mappings, the frontend can describe the pre-relayout and post-relayout states as an explicit `S1 -> SHUFFLE -> S2` operation. The normative contract and fallback requirements are defined in [LX Relayout SHUFFLE](LX-Relayout-SHUFFLE.md).
* Symbolic information
  * Link dsc dimensions to symbols
    * `std::map<PrimaryDimTypes, std::vector<VariableSymbol>> dimToSymbolMapping_` in `sdsc.dscs_[0]`
      * only fill one VariableSymbol per dimension
      * `VariableSymbol` should be a value coming from class `VariableDefinition`
  * in each datastage, fill max value and granularity for symbolic dimensions
    * `std::map<PrimaryDimTypes, SymbolicDimInfo> symbolicDimInfo_` in DataStructDims
  * if a symbolic dimension is divided across cores:
    * number of slices must be a divisor of granularity
    * max and granularity in datastages should be scaled accordingly
    * start addresses per core will need to be symbolic
      * in AllocateNode fill `FoldManager<int64_t> startAddressCoreCorelet_` with VariableSymbol entries
      * set `bool isStartAddrSymbolic_`
* Constants
  * if dataflow requires a constant value to be provided by frontend, a constantInfo entry is needed
  * `std::map<int, dsc2::ConstantInfo> constantInfo_` in `sdsc.dscs_[0]`. Key is irrelevant. Fields to fill:
    * `std::string name_` as agreed with ddl for that operation
    * `DataFormats dataFormat_`
    * `FoldManager<std::vector<uint32_t>> data_`  single constant value in binary format encoded as the dataformat specified above
      * do not replicate the binary encoding to fill the 32 bits if the data format is smaller
      * only fill one entry in the vector
      * first fold is for cores, set as Const fold type if same for all cores, set as Map if value changes across cores (very unlikely)
      * second fold is for corelets, set as Const fold type

### Supported OpFuncs in `sdsc.json`

OpFuncs are specified within sdsc.json as field `OpFuncs opFuncName` in `sdsc.dscs_[0].computeOp_[0]`.

| Category | OpFunc enum | OpFunc string | Op Precision | Constants | Notes
| --- | --- | --- | -- | -- | -- |
|  Matmul | BATCHMATMUL_FP8_FWD |  "batchmatmulfp8" | Inputs: FP8<1,4,3>, Output: DF16
|         | BATCHMATMUL_FWD |   "batchmatmul" | DF16
|         | BATCHMATMUL_INT4_FWD  |   "batchmatmulint4" | Inputs: INT4, Output: DF16
|         | BATCHMATMUL_INT8_FWD  |   "batchmatmulint8" | Inputs: INT8, Output: DF16 (scaled)
| Convolution | CONV2D_FP8_FWD  |   "conv2dfp8" | Inputs: FP8<1,4,3>, Output: DF16
|         | CONV2D_FWD  |   "conv2d" | DF16
|         | CONV2D_INT4_FWD  |   "conv2dint4" | Inputs: INT4, Output: DF16
|         | CONV2D_INT8_FWD  |   "conv2dint8" | Inputs: INT8, Output: DF16 (scaled)
|  Broadcast    |  ADD  |   "add"  | DF16 or FP32 |   | Broadcast supported on any number of dimensions and on one or both inputs
|         | BATCHNORM_FWD  |   "batchnormfwd" | DF16 or FP32 
|         | BIASADD  |   "biasadd" | DF16 or FP32 
|         | EQUAL  |   "equal" | DF16 or FP32 
|         | FNMS  |   "fnms" | DF16 or FP32 
|         | GREATEREQUAL  |   "greaterequal" | DF16 or FP32 
|         | LAYERNORM_NORM  |   "layernormnorm" | DF16 or FP32 
|         | LESSEREQUAL  |   "lesserequal" | DF16 or FP32 
|         | MAXIMUM  |   "maximum" | DF16 or FP32 
|         | MINIMUM  |   "minimum" | DF16 or FP32 
|         | MUL  |   "mul" | DF16 or FP32 
|         | NOTEQUAL  |   "notequal" | DF16 or FP32 
|         | REALDIV  |   "realdiv" | DF16 or FP32 
|         | REVSUB  |   "revsub" | DF16 or FP32 
|         | SUB  |   "sub" | DF16 or FP32 
|         | WHERE3  |   "where3" | DF16 or FP32 
|  Unary  | ABS  |   "abs" | DF16 or FP32 
|         | CLIP_FWD  |   "clip"  |  DF16 or FP32 |  `clipMin`, `clipMax`: minimum and maximum values to clip at
|         | EXP_FWD  |   "exp"  | DF16 or FP32 
|         | FAST_EXP_FWD  |   "fastexp" | DF16
|         | FAST_SIGMOID_FWD  |   "fastsigmoid" | DF16
|         | FLOOR  |   "floor" | DF16 or FP32
|         | GELU_FWD  |   "gelufwd" | DF16
|         | IDENTITY  |   "identity" | DF16 or FP32 
|         | LAYERNORM_SCALE  |  "layernormscale"  | DF16 or FP32 | `eps`: a small value added to the denominator in the calculation of layernorm for numerical stability
|         | LEAKYRELU_FWD  |  "leakyrelufwd" | DF16
|         | LOG_FWD  |   "log" | DF16
|         | MISH_FWD  |   "mish" | DF16
|         | NEG  |   "neg" | DF16 or FP32 
|         | RECIPROCAL  |   "reciprocal" | DF16 or FP32 
|         | RELU_FWD  |   "relufwd" | DF16 or FP32 
|         | RELU6_FWD  |   "relu6fwd" | DF16
|         | RSQRT  |   "rsqrt" | DF16
|         | SIGMOID_FWD  |   "sigmoid" | DF16 or FP32 
|         | SOFTPLUS  |   "softplus"  | DF16 | `softplusBeta`: value for the Softplus formulation <br>`softplusThresh`: values above this revert to a linear function
|         | SILU_FWD  |   "silu" | DF16 or FP32 
|         | SQRT_FWD  |   "sqrt" | DF16
|         | TANH_FWD  |   "tanh" | DF16
|  Reduction  | ABSMAX_NONSTICK  |   "absmaxnonstick" | DF16 or FP32 
|         | ABSMAX  |   "absmax" | DF16 or FP32 
|         | EXX2_ZEROMEAN  |   "exx2_zeromean" | DF16 or FP32 
|         | EXX2  |   "exx2"  | DF16 or FP32  | `exx2scale` 
|         | MAX_NONSTICK  |   "maxnonstick" | DF16 or FP32 
|         | MAX  |   "max" | DF16 or FP32 
|         | MEAN_NONSTICK  |   "meannonstick" | DF16 or FP32 | `scaling_factor`: reciprocal of the number of elements that are reduced, combined across all the reduction dimensions
|         | MEAN  |   "mean"  | DF16 or FP32 | `scaling_factor`: reciprocal of the number of elements that are reduced, combined across all the reduction dimensions
|         | MIN_NONSTICK  |   "minnonstick" | DF16 or FP32 
|         | MIN  |   "min" | DF16 or FP32 
|         | QUANT_SCALE_PER_TOKEN_FP8  |   "quantscalepertokenfp8"  | DF16 | `clipMin`, `clipMax`: minimum and maximum values to clip abs(input) distribution, typically smallest and largest positive values in the unquantized (input tensor) datatype <br>`mulConst`: reciprocal of the maximum value in the quantized datatype (for FP8<1,4,3> this is `1/448`)
|         | QUANT_SCALE_PER_TOKEN  |   "quantscalepertoken" | DF16
|         | SUM_NONSTICK  |   "sumnonstick" | DF16 or FP32 
|         | SUM  |   "sum" | DF16 or FP32 
|  Pooling  | AVGPOOL_FWD  |   "avgpoolfwd" | DF16 | `nmap`: reciprocal of the product of kernel size (`1/(kh*kw)`)
|         | AVGPOOL_NMAP_FWD |   "avgpoolnmapfwd" | DF16 
|         | DEPTHWISE_CONV_FWD  |   "depthwiseconv2dnative" | DF16 
|         | MAXPOOL_FWD  |   "maxpoolfwd" | DF16
| Scan    | MASK_BY_INDEX  |   "maskbyindex" | DF16 or FP32 
|         | TOPK_INDEX  |   "topkindex" | DF16 or FP32 
|         | TOPK_VALUE  |   "topkvalue" | DF16 or FP32 
| Quantization | CSQ_INT4_WT  |   "csqint4wt"   | Inputs: DF16 Output: INT4 |  `scaleact`: pre-quantization scale factor <br>`shiftact`: pre-quantization offset  |  Apply scale and shift to DL16 and quantize to INT4. Pack elements from four input sticks in a dimension different from the input stick dimension, alternating after every element
|         | CSQ_INT4  |   "csqint4"  | Inputs: DF16 Output: INT4  |  `scaleact`: pre-quantization scale factor <br>`shiftact`: pre-quantization offset  |  Apply scale and shift to DL16 and quantize to INT4. Alternating every 8 elements, pack elements from four input sticks: first two sticks in the same dimension as the input stick dimension, then two such groups taken across a dimension different from the input stick dimension
|         | CSQ_INT8_CH  |   "csqint8ch"  | Inputs: DF16 Output: INT8  |  `scaleact`: pre-quantization scale factor <br>`shiftact`: pre-quantization offset  |  Apply scale and shift to DL16 and quantize to INT8. Pack elements from two input sticks in the same dimension as the input stick dimension, alternating every 8 elements
|         | CSQ_INT8_MB  |   "csqint8mb"  | Inputs: DF16 Output: INT8 |  `scaleact`: pre-quantization scale factor <br>`shiftact`: pre-quantization offset  |  Apply scale and shift to DL16 and quantize to INT8. Pack elements from two input sticks in a dimension different from the input stick dimension, alternating  every 8 elements
|         | CSQ_INT8_WT  |   "csqint8wt"  | Inputs: DF16 Output: INT8 |  `scaleact`: pre-quantization scale factor <br>`shiftact`: pre-quantization offset  |  Apply scale and shift to DL16 and quantize to INT8. Pack elements from two input sticks in a dimension different from the input stick dimension, alternating after every element
|         | DL16TOFP32  |   "dl16tofp32"   | Input: DF16 Output: FP32 |  |  Convert DL16 to FP32. For every stick of input, two sticks will be produced
|         | FP32TODL16  |   "fp32todl16"   | Input: FP32 Output: DF16 |  |  Quantize FP32 to DL16. Pack elements from two input sticks in the same dimension as the input stick dimension, alternating every 8 elements
|         | FP8TODL16  |   "fp8todl16"   | Input: HFP8 Output: DF16 |  |  Convert FP8<1,4,3> to DL16. For every stick of input, two sticks will be produced
|         | Q_FP8_CH  |   "qfp8ch"   | Input: DF16 Output: FP8<1,4,3> |  |  Quantize DL16 to FP8<1,4,3>. Pack elements from two input sticks in the same dimension as the input stick dimension, alternating every 8 elements
|         | Q_FP8_MB  |   "qfp8mb"   | Input: DF16 Output: FP8<1,4,3> |  |  Quantize DL16 to FP8<1,4,3>. Pack elements from two input sticks in a dimension different from the input stick dimension, alternating every 8 elements
|         | Q_FP8_WT  |   "qfp8wt"   | Input: DF16 Output: FP8<1,4,3> |  |  Quantize DL16 to FP8<1,4,3>. Pack elements from two input sticks in a dimension different from the input stick dimension, alternating after every element
| Data shuffle | SHUFFLE | "shuffle" | Input and output formats must match | | Copy values between explicit static affine LX distributions without arithmetic. See [LX Relayout SHUFFLE](LX-Relayout-SHUFFLE.md).
| Stick Altering Data shuffle | ReStickifyOpHBM | "ReStickifyOpHBM" | DF16 | | Change the stick composition from one dimension to another dimension. Only one dimension is allowed in input and output stick layouts.


### Stick constraints for the operations


Each class of operation imposes constraints on the stick composition of its constituent tensors restricting which dimension can be present in the stick. Tensors will need to be padded to meet the stick constraints. There are noconstraints on the tensor layout beyond a stick.

Note: Stick constraints in an operation can cause a ripple effect---a tensor may need to padded even in its non-stick dimension because that dimension appers in the stick of another tensor feeding to the same operation. This is needed to ensure span of a dimension is consistent across all tensors.

#### BatchMatmul
The BatchMatmul op takes 2 inputs (Input1, Input2) and produces an output (Output1). It has 4 types of semantic dimensions:
* `reduction_dim`: Dimension that is present in Input1, Input2 and NOT in Output1. There can be only be a single dimension in this category. Note: this dimension gets reduced as part of the dot-product.
* `generated_dim`: Dimension that is present in Input2, Output1 and NOT in Input1. There can be only be a single dimension in this category.
* `preserved_dim`: Dimension that is present in Input1 and Output1 and NOT in Input2. There can be upto 2 dimensions in this category.
* `noreuse_dim`: Dimension that is present in all tensors - Input1, Input2 and Output1. There can be upto 2 dimensions in this category.

The following are the stick constraints that different precisons.
* Output1 tensor:
  * Stick composed of [`generated_dim`=64]. Note: Output1 is always in DF16 precision
* Input1 tensor:
  * DF16: Stick composed of [`reduction_dim`=64]
  * FP8/INT8: Stick composed of [`reduction_dim`=128]
  * INT4: Stick composed of 2 dimensions as: [`reduction_dim`=16, `preserved_dim`=2, `reduction_dim`=8]. Note: total of 256 elements in each stick.
* Input2 tensor:
  * DF16: Stick composed of [`generated_dim`=64]
  * FP8/INT8: Stick composed of [`reduction_dim`=2, `generated_dim`=64]
  * INT4: Stick composed of [`reduction_dim`=4, `generated_dim`=64]

Note: In Matmul, Input2 must also be padded along `reduction_dim` (which is not in its stick). This is because `reduction_dim` is part of the stick of Input1 and Input2 therefore needs to be padded for their dimension spans to be consistent.

#### Convolution
Same as matmul with the only difference of the INT4 Input1 stick layout: [`reduction_dim`=16, `W`=2, `reduction_dim`=8], where `W` is the width in pixels according to `NHWC` notation.

#### Reduction
For sum/max/min/mean/absmax/exx2:
* the reduction dimension should be the only dimension in the stick
* same stick layout in input and output (output will have scale=-2 for reduced dimension)

For sum-nonstick/max-nonstick/min-nonstick/mean-nonstick/absmax-nonstick (there is no nonstick version of exx2):
* any number of non-reduction dimensions in the stick is allowed
* same stick layout in input and output

#### Unary and Broadcast operations
Any stick layout is acceptable, but all inputs and output must have same stick layout.
If a stick dimension has broadcast in a tensor, all stick dimensions of that tensor must have broadcast.
If a stick dimension has broadcast in all tensors (inputs and output), then its size in the SDSC must be set to the number of elements that one stick would have if that dimension actually existed.

#### Scan

In top-k, neither the reduction dimension nor k can be in the stick, any number of other dimensions can be in the stick

#### layernormscale/layernormnorm/exx2
Stick should only have the normalization dimension in it

#### Pooling
Window dimensions not allowed in the stick, any number of other dimensions can be in the stick

#### Quantization operations
For all down-casting operations
* the input should always have only one dimension in the stick (DL16: [`inpdim`=64], FP32: [`inpdim`=32])
* for the `wt` family of quantizations, output stick should have one more dimension innermost (INT8/FP8: [`otherdim`=2,`inpdim`=64], INT4: [`otherdim`=4,`inpdim`=64])
* for the `mb` family of quantizations, output stick should have one more dimension inserted at the slice level (INT8/FP8: [`inpdim`=8, `otherdim`=2, `inpdim`=8], INT4: [`inpdim`=16, `otherdim`=2, `inpdim`=8])
* for the `ch` family of quantizations, output stick still only has one dimension, just more elements (INT8/FP8: [`inpdim`=128])

For all up-casting operations, both input and output should have the same only one dimension in the stick.

#### Stick Altering Data shuffle
Restickify can only operate on input sticks with elements from a single dimension d1 and produce output sticks with elements from a single dimension d2. There is no restriction on which d1 and d2 dimensions are picked.

### Core work division constraints for the operations
For all operations, any constituent dimension is allowed to be split across cores. The following constraints apply to the work assigned per core:
* For all data tensors, The work assigned must be multiple of stick size in any given dimension. This restriction does not apply to index tensors used for indirect access.
* The span of addresses accessed from DDR for any given tensor must not exceed 256MB.
* For all index tensors used for indirect access, for each dimension present in the stick, the work assigned to a core should span an intergral number of sticks or <1 stick.

When operation involves reduction across multiple dimensions, only one of its reduction dimensions is allowed to be split across cores. There is no constraint on operations with a single reduction dimension.

## Examples

Multiple examples in increasing order of complexity are available [here](examples/).

## **Metrics **
* Ability to express all torch operators that are mappable to AIU (post-inductor transformations and decompositions)
* Ability to express desired computation mapping across cores for each operation
