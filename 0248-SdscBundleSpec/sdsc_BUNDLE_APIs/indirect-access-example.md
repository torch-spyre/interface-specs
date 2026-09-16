# Indirect Access Example — JSON (Top-K Gather Operation)

Indirect access (also called *paged access*) allows a compute operation to read values
from a tensor whose elements are not contiguous in memory; instead, a second tensor
supplies an index (or precomputed byte address) for each element to be read.  The
canonical use case is a **Top-K gather**: an index tensor holding the positions of the
top-K elements, and a value tensor holding the full data buffer from which those elements
are gathered.

This example walks through a single-core `topkvalue` operation over a 1-D tensor of 512
`SEN169_FP16` elements, selecting the top 8.  The value tensor lives in HBM and the index
tensor (type `KERNEL_IDX`) also lives in HBM.

---

## What changes for indirect access

Compared with a direct-access allocation, three extra things must be done:

| What | Where | Value |
|---|---|---|
| Mark the value tensor's allocate node | `ScheduleTreeNode.indirectAllocType_` | `"value_tensor"` |
| Mark the index tensor's allocate node | `ScheduleTreeNode.indirectAllocType_` | `"index_tensor"` |
| Cross-link the two nodes | `ScheduleTreeNode.relatedIndirectAccessAlloc_` | name of the **other** node |
| Declare how to interpret the index | `ScheduleTreeNode.indexTensorType_` (index node only) | `"index"` or `"address"` |
| Set the page size on the value tensor | `ScheduleTreeNode.maxDimSizes_` | page size in elements |
| Reference the index tensor in the compute op | `ComputeOperation.indirectAccessIndexLabeledDs` | composite name string |

The value tensor uses `dsType_: "KERNEL"` in `labeledDs_`; the index tensor uses
`dsType_: "KERNEL_IDX"`.

---

## Complete JSON

```json
{
  "topk_gather": {
    "coreFoldProp_":    {"factor_": 1, "label_": "core"},
    "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
    "numCoresUsed_": 1,
    "coreIdToDsc_": {
      "0": 0
    },
    "coreIdToDscSchedule": {
      "0": [[0, 0, 0, 0]]
    },
    "dscs_": [
      {
        "topk": {
          "numCoresUsed_": 1,
          "coreIdsUsed_": [0],

          "N_": {
            "in_": 512,
            "out_": 8
          },

          "primaryDsInfo_": {
            "INPUT": {
              "layoutDimOrder_": ["in"],
              "stickDimOrder_":  ["in"],
              "stickSize_":      [64]
            },
            "OUTPUT": {
              "layoutDimOrder_": ["out"],
              "stickDimOrder_":  ["out"],
              "stickSize_":      [64]
            },
            "KERNEL": {
              "layoutDimOrder_": ["in"],
              "stickDimOrder_":  ["in"],
              "stickSize_":      [64]
            },
            "KERNEL_IDX": {
              "layoutDimOrder_": ["out"],
              "stickDimOrder_":  ["out"],
              "stickSize_":      [64]
            }
          },

          "dataStageParam_": {
            "0": {
              "name_": "core",
              "ss_": {"in_": 512, "out_": 8},
              "el_": {"in_": 512, "out_": 8}
            }
          },

          "scheduleTree_": [
            {
              "nodeType_":   "allocate",
              "name_":       "input_alloc",
              "ldsIdx_":     0,
              "component_":  "hbm",
              "layoutDimOrder_": ["in"],
              "maxDimSizes_":    [-1],
              "indirectAllocType_": "no_indirection",
              "startAddressCoreCorelet_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}],
                "dim_prop_attr": [
                  {"factor_": 1, "label_": "core"},
                  {"factor_": 1, "label_": "corelet"}
                ],
                "data_": {"[0, 0]": "0"}
              },
              "coordinates_": {
                "coordInfo": {
                  "in": {
                    "spatial":  3,
                    "temporal": 0,
                    "elemArr":  2,
                    "padding":  "nopad",
                    "folds": {
                      "dim_prop_func": [
                        {"Affine": {"alpha_": 512, "beta_": 0}},
                        {"Affine": {"alpha_": 0,   "beta_": 0}},
                        {"Affine": {"alpha_": 0,   "beta_": 0}},
                        {"Affine": {"alpha_": 1,   "beta_": 0}},
                        {"Affine": {"alpha_": 1,   "beta_": 0}}
                      ],
                      "dim_prop_attr": [
                        {"factor_": 1,  "label_": "core_fold"},
                        {"factor_": 1,  "label_": "corelet_fold"},
                        {"factor_": 1,  "label_": "row_fold"},
                        {"factor_": 64, "label_": "elem_arr_0"},
                        {"factor_": 8,  "label_": "elem_arr_1"}
                      ]
                    }
                  }
                }
              }
            },

            {
              "nodeType_":   "allocate",
              "name_":       "output_alloc",
              "prev_":       "input_alloc",
              "ldsIdx_":     1,
              "component_":  "hbm",
              "layoutDimOrder_": ["out"],
              "maxDimSizes_":    [-1],
              "indirectAllocType_": "no_indirection",
              "startAddressCoreCorelet_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}],
                "dim_prop_attr": [
                  {"factor_": 1, "label_": "core"},
                  {"factor_": 1, "label_": "corelet"}
                ],
                "data_": {"[0, 0]": "8192"}
              },
              "coordinates_": {
                "coordInfo": {
                  "out": {
                    "spatial":  3,
                    "temporal": 0,
                    "elemArr":  2,
                    "padding":  "nopad",
                    "folds": {
                      "dim_prop_func": [
                        {"Affine": {"alpha_": 8, "beta_": 0}},
                        {"Affine": {"alpha_": 0, "beta_": 0}},
                        {"Affine": {"alpha_": 0, "beta_": 0}},
                        {"Affine": {"alpha_": 1, "beta_": 0}},
                        {"Affine": {"alpha_": 1, "beta_": 0}}
                      ],
                      "dim_prop_attr": [
                        {"factor_": 1,  "label_": "core_fold"},
                        {"factor_": 1,  "label_": "corelet_fold"},
                        {"factor_": 1,  "label_": "row_fold"},
                        {"factor_": 8,  "label_": "elem_arr_0"},
                        {"factor_": 1,  "label_": "elem_arr_1"}
                      ]
                    }
                  }
                }
              }
            },

            {
              "nodeType_":   "allocate",
              "name_":       "value_alloc",
              "prev_":       "output_alloc",
              "ldsIdx_":     2,
              "component_":  "hbm",
              "layoutDimOrder_": ["in"],
              "maxDimSizes_":    [64],
              "indirectAllocType_":          "value_tensor",
              "relatedIndirectAccessAlloc_": "index_alloc",
              "startAddressCoreCorelet_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}],
                "dim_prop_attr": [
                  {"factor_": 1, "label_": "core"},
                  {"factor_": 1, "label_": "corelet"}
                ],
                "data_": {"[0, 0]": "16384"}
              },
              "coordinates_": {
                "coordInfo": {
                  "in": {
                    "spatial":  3,
                    "temporal": 0,
                    "elemArr":  2,
                    "padding":  "nopad",
                    "folds": {
                      "dim_prop_func": [
                        {"Affine": {"alpha_": 512, "beta_": 0}},
                        {"Affine": {"alpha_": 0,   "beta_": 0}},
                        {"Affine": {"alpha_": 0,   "beta_": 0}},
                        {"Affine": {"alpha_": 1,   "beta_": 0}},
                        {"Affine": {"alpha_": 1,   "beta_": 0}}
                      ],
                      "dim_prop_attr": [
                        {"factor_": 1,  "label_": "core_fold"},
                        {"factor_": 1,  "label_": "corelet_fold"},
                        {"factor_": 1,  "label_": "row_fold"},
                        {"factor_": 64, "label_": "elem_arr_0"},
                        {"factor_": 8,  "label_": "elem_arr_1"}
                      ]
                    }
                  }
                }
              }
            },

            {
              "nodeType_":   "allocate",
              "name_":       "index_alloc",
              "prev_":       "value_alloc",
              "ldsIdx_":     3,
              "component_":  "hbm",
              "layoutDimOrder_": ["out"],
              "maxDimSizes_":    [-1],
              "indirectAllocType_":          "index_tensor",
              "relatedIndirectAccessAlloc_": "value_alloc",
              "indexTensorType_":            "index",
              "startAddressCoreCorelet_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}],
                "dim_prop_attr": [
                  {"factor_": 1, "label_": "core"},
                  {"factor_": 1, "label_": "corelet"}
                ],
                "data_": {"[0, 0]": "24576"}
              },
              "coordinates_": {
                "coordInfo": {
                  "out": {
                    "spatial":  3,
                    "temporal": 0,
                    "elemArr":  2,
                    "padding":  "nopad",
                    "folds": {
                      "dim_prop_func": [
                        {"Affine": {"alpha_": 8, "beta_": 0}},
                        {"Affine": {"alpha_": 0, "beta_": 0}},
                        {"Affine": {"alpha_": 0, "beta_": 0}},
                        {"Affine": {"alpha_": 1, "beta_": 0}},
                        {"Affine": {"alpha_": 1, "beta_": 0}}
                      ],
                      "dim_prop_attr": [
                        {"factor_": 1,  "label_": "core_fold"},
                        {"factor_": 1,  "label_": "corelet_fold"},
                        {"factor_": 1,  "label_": "row_fold"},
                        {"factor_": 8,  "label_": "elem_arr_0"},
                        {"factor_": 1,  "label_": "elem_arr_1"}
                      ]
                    }
                  }
                }
              }
            }
          ],

          "labeledDs_": [
            {
              "ldsIdx_":     0,
              "dsName_":     "topk-Input",
              "dsType_":     "INPUT",
              "scale_":      [1.0],
              "wordLength":  2,
              "dataFormat_": "SEN169_FP16",
              "memOrg_": {
                "hbm": {"isPresent": 1},
                "lx":  {"isPresent": 0}
              }
            },
            {
              "ldsIdx_":     1,
              "dsName_":     "topk-Output",
              "dsType_":     "OUTPUT",
              "scale_":      [1.0],
              "wordLength":  2,
              "dataFormat_": "SEN169_FP16",
              "memOrg_": {
                "hbm": {"isPresent": 1},
                "lx":  {"isPresent": 0}
              }
            },
            {
              "ldsIdx_":     2,
              "dsName_":     "topk-Values",
              "dsType_":     "KERNEL",
              "scale_":      [1.0],
              "wordLength":  2,
              "dataFormat_": "SEN169_FP16",
              "memOrg_": {
                "hbm": {"isPresent": 1},
                "lx":  {"isPresent": 0}
              }
            },
            {
              "ldsIdx_":     3,
              "dsName_":     "topk-Indices",
              "dsType_":     "KERNEL_IDX",
              "scale_":      [1.0],
              "wordLength":  4,
              "dataFormat_": "IEEE_INT32",
              "memOrg_": {
                "hbm": {"isPresent": 1},
                "lx":  {"isPresent": 0}
              }
            }
          ],

          "constantInfo_": "{}",

          "computeOp_": [
            {
              "exUnit":      "sfp",
              "opFuncName":  "topkvalue",
              "attributes_": {
                "dataFormat_": "SEN169_FP16",
                "fidelity_":   "regular"
              },
              "location": "Inner",
              "inputLabeledDs":               ["topk-Input-idx0"],
              "outputLabeledDs":              ["topk-Output-idx1"],
              "indirectAccessIndexLabeledDs": ["topk-Indices-idx3"]
            }
          ]
        }
      }
    ]
  }
}
```

---

## Annotation

### The four tensor roles

| `ldsIdx_` | `dsName_` | `dsType_` | Role |
|---|---|---|---|
| 0 | `topk-Input` | `INPUT` | The full 512-element source buffer that Top-K scans |
| 1 | `topk-Output` | `OUTPUT` | The 8-element result buffer holding the gathered values |
| 2 | `topk-Values` | `KERNEL` | The paged value buffer (same data as input in this example; marked `value_tensor`) |
| 3 | `topk-Indices` | `KERNEL_IDX` | The 8-element index buffer holding positions in `topk-Values`; marked `index_tensor` |

### Indirect-access fields

**`value_alloc` node** — marks the data buffer that is accessed non-contiguously:

- `indirectAllocType_: "value_tensor"` — this allocation is the data side of an indirect pair.
- `relatedIndirectAccessAlloc_: "index_alloc"` — cross-link to the index node.
- `maxDimSizes_: [64]` — page size: the backend treats the value buffer as a series of
  64-element pages; the index tensor provides which page each gathered element comes from.

**`index_alloc` node** — marks the tensor whose elements are used as look-up keys:

- `indirectAllocType_: "index_tensor"` — this allocation supplies indices.
- `relatedIndirectAccessAlloc_: "value_alloc"` — cross-link back to the value node.
- `indexTensorType_: "index"` — elements are element indices (not precomputed byte
  addresses). Use `"address"` when the index tensor instead holds device byte addresses.

**`ComputeOperation`** — references the index tensor separately from input/output:

- `indirectAccessIndexLabeledDs: ["topk-Indices-idx3"]` — the backend uses this list
  to know which tensor drives the indirect look-up, distinct from the primary inputs
  and outputs.

### Memory layout

| Tensor | HBM start | Size |
|---|---|---|
| `topk-Input` (512 × FP16) | `0` | 1 024 B |
| `topk-Output` (8 × FP16) | `8192` | 16 B |
| `topk-Values` (512 × FP16, page 64) | `16384` | 1 024 B |
| `topk-Indices` (8 × INT32) | `24576` | 32 B |

---

| [← Previous: Complete Example (JSON)](complete-example.md) | [↑ Table of Contents](README.md) | [Next: Reference →](reference.md) |
|:--|:--:|--:|
