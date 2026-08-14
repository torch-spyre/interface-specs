# SDSC JSON API

## Introduction

SuperDSC or SDSC is a json representation of an operation to be performed by the Spyre backend, DeepTools.
It also includes specifications of the input and output tensors needed by the operation, and how those
tensors are to be mapped onto the various cores of the Spyre device (that is, core mappings).

The different sections of an SDSC JSON file describe the following:

- Operation type and attributes
- Input and output tensor configurations
- Tensor layouts and stick configurations
- Memory organization and data staging
- Work division across cores

## Key Components

The SDSC JSON structure includes:

1. **SuperDSC Object** — Root container for the operation
2. **DesignSpaceConfig** — Core configuration for computation
3. **DataStageParam** — Tensor data staging information
4. **ScheduleTreeNode** — Execution schedule
5. **ComputeOperation** — Operation type and attributes
6. **MemoryOrganization** — Tensor layout and memory placement

The above components are specified in the SDSC JSON file using a few top-level fields and an array of structures termed `dscs_[]` (design space configs).
Each `dsc_` entry consists of some leaf (final) fields and a few composite (non-leaf) ones that can be drilled down into.

## Object Hierarchy

See [JSON-object-Hierarchy.md](JSON-object-Hierarchy.md) for the full hierarchy diagram.

## SuperDsc Object

See [superdsc-object.md](superdsc-object.md) for the complete field reference.

## FoldProperty

See [foldproperty.md](foldproperty.md) for details.

## FoldManager

See [foldmanager.md](foldmanager.md) for details.

## SuperDSC Object Fields

The SuperDSC object is the root container that encapsulates all information needed to execute an operation on the Spyre backend.

### Core Metadata and Folding information

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 1 | `<_name>`, e.g., `"0_batchmatmul"` | Identifier for the sdsc | |
| 2 | `numCoresUsed_` | Number of cores used for the operation | |
| 3 | `sdscFoldProps_` | Denotes folds over time | Vector of length 1 containing fields **factor_** and **label_**, set to 1 and "time", resp., as in `[{"factor":1, "label_": "time"}]`. |
| 4 | `sdscFolds_` | Contains sub-fields **dim_prop_func** and **dim_prop_attr**, describing folds over time. | Refer to default values in sample sdscs, which are sufficient in most of the cases. |
| 5 | `coreFoldProp_` | Denotes folding over cores using sub-fields **factor** and **label_** | factor=number of cores used and label='core' |
| 6 | `coreletFoldProp_` | Denotes folding over corelets using sub-fields **factor** and **label_** | factor=1 or 2 and label='corelet' |
| 7 | `fold_coord_` | For a folded SDSC with multiple fold types, fold_coord_ stores the index at each fold dimension. Specifies which unfolded variant an SDSC represents by storing its coordinate indices in folded space. | |

### Dimensions and Work Slicing

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 8 | `N_` | List of all dimensions across tensors used by ops in all dsc_'s in the sdsc. Contains each dimension's size across all cores. For dimensions that are padded (such as convolution's image dimensions) padding details are also included. See [Padding](padding.md). | |
| 9 | `numWkSlicesPerDim_` | A map keyed by the dimension name indicating the number of slices into which each dimension is split. | The product of the slices over all dimensions should equal the number of cores (across which the operation is executed). |
| 10 | `coreIdToWkSlice_` | Map from core id to the slice of each dimension assigned to it. | Nested map. Outer key is core id. Inner key is dimension name. The slice number assigned to a core ranges from 0 to the total slice count for the dim indicated by `numWkSlicesPerDim_`. |
| 11 | `coreIdToDsc_` | Mapping from core id to dsc number when the sdsc contains multiple dsc's. | |
| 12 | `ldsShareInfo` | LabeledDS sharing information. Tracks tensor sharing across multiple DSC instances. | |
| 13 | `opFuncsUsed_` | | |
| 14 | `prodConsList_` | | |

### Scheduling Related

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 15 | `coreIdToDscSchedule_` | Vector of DscScheduleStep schedule steps specified for each core. Defines the execution sequence of operations on each core by specifying which data and dl dsc operations execute on this core, the order they execute in, and data-flow dependencies. | |
| 16 | `target_` | Specifies the target backend/hardware platform for executing the SDSC op. | Can be one of SENTIENT, SENULATOR, SENPCFG, SENTF, SYSTEMC, R5SS, HOST |

### Nested Data Structures

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 17 | `dscs_` | Array of DesignSpaceConfig (or DataStageConfig) structures. | Vector to express core work mapping for an operation. With balanced work division, only one entry in the vector is needed. |

## DesignSpaceConfig Fields

Field `dscs_` in the top-level SuperDSC object specifies the design space configuration for the operation.

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 1 | `N_` | Similar to SuperDSC.N_ above. Details of dimensions used by a specific dsc_. | In addition to the fields of SuperDSC.N_, padding details are provided in sub-field `paddingSizes_`. One entry per dimension that is padded. See [Padding](padding.md) for details. |
| 2 | `coreIdsUsed_` | List of cores used by the ops in this dsc. | Specified as a list of core ids, e.g., [0, 6, 10]. |
| 3 | `dataStageParam_` | Specifies sizes per dimension for each core, in **steady state** and **epilogue** stages. When work is not uniformly divided across time steps, the work done (in terms of number of elements per dimension) can be different in the final stage, which is referred to the epilogue stage. | The overall field has "0" as the key. Has two nested entries, keyed using "ss_" and "el_" denoting steady-state and epilogue, respectively. el_ will differ from ss_ only when work division across time steps is not uniform. Both ss_ and el_ fields include a "name_" sub-field with value set to "core_" and one sub-field for each dimension. Like N_, padding details for window/padded operations are added to padding sub-fields within dataStageParam_. See [Padding](padding.md) for more details. |
| 4 | `labeledDs_` | Vector, each of whose elements represents a physical tensor used in the dsc_'s operation. Both input and output tensors need to be listed. | Nested fields of each element of labeledDs_ are explained in the labeledDs table below. |
| 5 | `primaryDsInfo_` | Defines DsTypes used in the dsc_. A DsType denotes a tensor type, which corresponds to a list of dimensions and the stick dimension(s). Multiple physical tensors can share a DsType. Currently defined types are **INPUT**, **OUTPUT**, **KERNEL**, **KERNEL_IDX**, and **NOT_SET**. | One top-level field within primaryDsInfo_ for each DsType. Each type contains sub-fields **layoutDimOrder_**, **stickDimOrder_**, and **stickSize_**. layoutDimOrder_ and stickDimOrder_ are vectors of dimension names. stickSize_ is a vector of integers whose length matches that of stickDimOrder_, with a 1-to-1 correspondence from the dimension in the latter to the size in the former. |
| 6 | `pdsRelation_` | | Has boolean `isPdsReuse` sub-field. |
| 7 | `dimToSymbolMapping_` | | |
| 8 | `constantInfo_` | | |
| 9 | `scheduleTree_` | Schedule of computations | Sub-fields explained in ScheduleTree table below. |
| 10 | `computeOp_` | Describes the compute operations performed by the dsc_ of the SDSC. More than one op will be specified in case of fused operations. | Explained in Compute Op table below. |

## Tensor Description (LabeledDSInfo) Fields

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 1 | `ldsIdx` | Index number | |
| 2 | `dsName` | A distinct name for the tensor. Used to identify the input and output tensors associated with a computeOp_. | The name of the tensor used in computeOp_ appends idx\<ldsIdx\> to the dsName in labeledDs_. E.g., A labeledDS_ with `"dsName_": "convolution-Tensor0"` and `"ldsIdx_": 0` is denoted as `"convolution-Tensor0-idx0"` in the computeOp_ section. |
| 3 | `dsType` | dsType of this tensor. Should be a type defined in primaryDsInfo_ of the parent _dsc. | |
| 4 | `scale_` | A vector with one entry per dimension present in the dsType of this tensor. | A scale of 1 is normal, -1 is reduced / broadcast, -2 is reduced / broadcast stick dimension. Order matches layoutDimOrder_ in primaryDsInfo_. |
| 5 | `wordLength_` | word length in number of bytes | |
| 6 | `dataFormat_` | One of: SEN169_FP16, IEEE_FP32, INVALID, SEN143_FP8, SEN152_FP8, SEN153_FP9, SENINT2, SENINT4, SENINT8, SENINT16, SENINT24, IEEE_INT64, IEEE_INT32, SENUINT32, SENUINT2, IEEE_FP16, BOOL, BFLOAT16, SEN18F_FP24, SEN080_FP8, SEN053_FP8, SEN121_FP4 | For Spyre, SEN169_FP16 is the most common. |
| 7 | `memOrg_` | Indicates memory residency (HBM vs. LX) via keys **hbm** and **lx**, which have nested sub-fields including **isPresent**, **isPadded**, **isZeroPadded**. | |

## ScheduleTree Node (scheduleTree_) Fields

ScheduleTree is a tree (list) of ScheduleNodes which could be of types BLOCK, LOOP, TRANSFER, COMPUTE, SYNC, CONDITION, ALLOCATE, STICKMASK (among others).

One ScheduleNode of type ALLOCATE needs to be added per tensor in LabeledDs. Only ALLOCATE nodes need to be filled from the front end.

### ALLOCATE node Fields

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 1 | `nodeType_` | Type of Schedule node | Always **allocate** |
| 2 | `name_` | An easy to identify name. | |
| 3 | `ldsIdx_` | Index into the tensor DS information in labeledDs_. | Indicates which labeledDs_ is being allocated. |
| 4 | `component_` | One of SenComponents such as HBM, LX to indicate memory residency | |
| 5 | `padding_` | Padding type applicable for each dimension of the tensor that is padded. | One of NOPAD, LOWERED_PADDED, PADDED_NOZEROPAD, PADDED_WZEROPAD, PADDED_FULLSPAN, and PADDED_FULLSPAN_WUNNEEDED. See [Padding](padding.md) for more details. |
| 6 | `layoutDimOrder_` | Order of dimensions on the device | Tensor layout specified from inner to outer. |
| 7 | `maxDimSizes_` | Maximum size of each dimension of the tensor. | Generally set to -1. |
| 8 | `isStartAddrSymbolic_` | 0 or 1 indicating whether the start addresses specified in `data_` field of `startAddressCoreCorelet_` are symbolic. | See [Folding](folding.md) |
| 8 | `startAddressCoreCorelet_` | Start address of a tensor per core. | See [Folding](folding.md) |
| 9 | `coordinates_` | Tensor coordinates per dimension. | See [Folding](folding.md) |
| 10 | `backGapCore_` | Records extra empty space ("back gap") added at the end of a dimension in an allocated buffer, tracked independently for each core (or HBM). | The field is of the form: `"backGapCore" : { <dim string> : { <"core id"> \| "-1" : back gap size as a string} ....}`. A key of "-1" indicates that the specification is for HBM. If it is a core id, then the back gap is for the LX scratchpad of the core. |
| 11 | `indirectAllocType_` | One of **no_indirection**, **index_tensor**, and **value_tensor** | **no_indirection** indicates an ordinary, directly-addressed allocation. **index_tensor** indicates the allocation holds indices used to look up into another tensor. **value_tensor** indicates the allocation holds the actual data that gets gathered/scattered via those indices. |
| 12 | `relatedIndirectAccessAlloc_` | Provides the name of the tensor (allocate node in schedule tree) that provides actual data for the indirect reference when `indirectAllocType_` is **index_tensor**. | Non-empty only when `indirectAllocType_` is **index_tensor**. |

## Compute Operation (computeOp_) Fields

Specifies details of compute operation to be performed on the tensors at the device.

| S.No. | Field Name | Purpose / Functionality | How to Fill |
|-------|-----------|------------------------|-------------|
| 1 | `exUnit_` | The exact compute unit on the device on which the operation is to be executed. | Typically one of PT (PT array), PE, and SFP (floating point unit). |
| 2 | `opFuncName_` | The name of the operation to be executed on the device. | |
| 3 | `attributes_` | Specifies attributes such as dataFormat_, fidelity_ | |
| 4 | `inputLabeledDs_` | Identifies tensors from labeledDs_ that form inputs to the operation. | |
| 5 | `outputLabeledDs` | Identifies tensor from labeledDs_ that forms the output of the operation. | |
| 6 | `indirectAccessIndexLabeledDs_` | | |
| 7 | `interimLabeledDs_` | | |

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: SuperDsc Object →](superdsc-object.md) |
|:--|:--:|--:|
