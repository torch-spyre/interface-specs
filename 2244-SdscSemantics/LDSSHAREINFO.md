# ldsShareInfo_ Field: SuperDsc Class Documentation

## Definition

**Location**: [superdsc.h]

```cpp
std::vector<LabeledDsShareInfo> ldsShareInfo_;
```

**Type**: `std::vector<LabeledDsShareInfo>` - Array of sharing information structures

---

## LabeledDsShareInfo Structure

**Location**: [dscdefn.h]

```cpp
struct LabeledDsShareInfo {
  std::string dsName_;  // Name of the LabeledDataStructure (tensor)

  std::vector<std::set<int>> sharedDscs;  // DSC indices sharing this LDS

  std::vector<std::vector<std::pair<std::string, int>>>
      repeatHbmLxThroImbalDim;  // indexed by DSCid - HBM/LX repetition info
};
```

---

## Purpose

The `ldsShareInfo_` field tracks **tensor sharing across multiple DesignSpaceConfig (DSC) instances** in scenarios where:

1. **Multiple work divisions** exist for the same operation (imbalanced work distribution)
2. **Multiple DSCs** process the same tensor but with different work splits
3. **Tensor dimensions** vary across DSCs due to work imbalance

### Example: Multi-DSC Operation

**Scenario**: ConvNet layer with imbalanced core utilization

| DSC Index | Cores | Output Batch (MB) | Input Dimension |
|-----------|-------|-------------------|-----------------|
| 0 | 0-1 | 8 | 64 |
| 1 | 2-3 | 4 | 32 |
| 2 | 4-5 | 2 | 16 |
| 3 | 6-7 | 1 | 8 |

All four DSCs process the same INPUT, KERNEL, and OUTPUT tensors but with different dimension slices.

**ldsShareInfo_** tracks:
- Which tensors are shared (INPUT, KERNEL shared by all 4 DSCs)
- Which DSCs share each tensor
- How dimensions repeat/map across DSCs

---

## Field Details

### 1. dsName_ (std::string)

**Purpose**: Identifies the LabeledDataStructure being tracked

**Example**:
```
"Tensor0" - input activation
"Tensor1" - kernel weights
"Tensor2" - output result
```

**Source**: Copied from `LabeledDsInfo.dsName_` in the DSC

---

### 2. sharedDscs (std::vector<std::set<int>>)

**Purpose**: **Specifies which DSCs share this tensor at each "level"** (typically by work split dimension)

**Structure**:
- Vector index = work split level (or dimension partition)
- Set value = DSC indices that share that partition

**Example: 4-DSC imbalanced operation**

```cpp
// KERNEL tensor: shared across all 4 DSCs in one chunk
sharedDscs = {{0, 1, 2, 3}};

// INPUT tensor: shared across DSCs but split by batch dimension
sharedDscs = {{0, 1}, {2, 3}};  // MB 0-7 in DSC 0,1 / MB 8-15 in DSC 2,3

// OUTPUT tensor: separate per DSC (each writes different output slice)
sharedDscs = {{0}, {1}, {2}, {3}};
```

**Interpretation**:
- **KERNEL** (`{0,1,2,3}`): Single entry → whole kernel shared by all
- **INPUT** (`{0,1}, {2,3}`): Two entries → tensor split into 2 partitions, each shared by subset
- **OUTPUT** (`{0}, {1}, {2}, {3}`): Four entries → each DSC owns separate output partition

**Code Example** (from dsi_imbawork_test.cpp:290-296):
```cpp
if (lds.dsType_ == DsTypes::KERNEL) {
  newLdsShare.sharedDscs = {{0, 1, 2, 3}};  // All 4 DSCs share kernel
} else if (lds.dsType_ == DsTypes::INPUT) {
  newLdsShare.sharedDscs = {{0, 1}, {2, 3}};  // Input split 2 ways
} else {  // OUTPUT
  newLdsShare.sharedDscs = {{0}, {1}, {2}, {3}};  // Output per DSC
}
newLdsShare.repeatHbmLxThroImbalDim.resize(4);
sdsc.ldsShareInfo_.push_back(newLdsShare);
```

---


## Key Insight

`ldsShareInfo_` is **primarily used for multi-DSC scenarios with imbalanced work distribution**, where:

1. **Multiple DesignSpaceConfigs** process the same operation (different work slices per core)
2. **Tensors are shared** across DSCs (kernel or input replication)
3. **Dimensions vary** between DSCs (due to different work partitioning)
4. **Transfer adjustments** needed when moving data between cores with different slices

For **single-DSC balanced operations** (the common case), ldsShareInfo_ is **empty** and not used.

---

## Related Files

- **dscdefn.h:787-792**: LabeledDsShareInfo struct definition
- **superdsc.h:65**: ldsShareInfo_ field in SuperDsc
- **superdsc.cpp:269-321**: Export to JSON
- **superdsc.cpp:678-722**: Import from JSON
- **sdscHelper.cpp:23-39**: Usage in reverse translation
- **dsi_imbawork_test.cpp:285-313**: Populate in multi-DSC test
