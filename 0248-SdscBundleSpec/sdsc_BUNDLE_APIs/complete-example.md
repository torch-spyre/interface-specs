# Complete Example — JSON (Simple GELU Operation)

```json
{
  "gelu_forward": {
    "coreFoldProp_": {
      "factor_": 2,
      "label_": "core"
    },
    "coreletFoldProp_": {
      "factor_": 2,
      "label_": "corelet"
    },
    "numCoresUsed_": 2,
    "coreIdToDsc_": {
      "0": 0,
      "1": 0
    },
    "numWkSlicesPerDim_": {
      "mb": 2,
      "out": 1
    },
    "coreIdToWkSlice_": {
      "0": {"mb": 0, "out": 0},
      "1": {"mb": 1, "out": 0}
    },
    "coreIdToDscSchedule": {
      "0": [[-1, 0, 0, 0]],
      "1": [[-1, 0, 0, 0]]
    },
    "dscs_": [
      {
        "gelu": {
          "numCoresUsed_": 2,
          "coreIdsUsed_": [0, 1],
          "N_": {
            "mb_": 32,
            "out_": 128
          },
          "dataStageParam_": {
            "0": {
              "name_": "core",
              "ss_": {"mb_": 16, "out_": 128},
              "el_": {"mb_": 16, "out_": 128}
            }
          },
          "primaryDsInfo_": {
            "INPUT": {
              "layoutDimOrder_": ["mb", "out"],
              "stickDimOrder_": ["out"],
              "stickSize_": [64]
            },
            "OUTPUT": {
              "layoutDimOrder_": ["mb", "out"],
              "stickDimOrder_": ["out"],
              "stickSize_": [64]
            }
          },
          "scheduleTree_": [
            {
              "nodeType_": "allocate",
              "name_": "input_alloc",
              "ldsIdx_": 0,
              "component_": "hbm",
              "layoutDimOrder_": ["mb", "out"],
              "maxDimSizes_": [32, 128],
              "startAddressCoreCorelet_": {
                "dim_prop_func": [{"Map": {}}, {"Const": {}}],
                "dim_prop_attr": [
                  {"factor_": 2, "label_": "core"},
                  {"factor_": 2, "label_": "corelet"}
                ],
                "data_": {
                  "[0, 0]": "0",
                  "[1, 0]": "8192"
                }
              },
              "coordinates_": {
                "coordInfo": {
                  "mb": {
                    "spatial":  3,
                    "temporal": 0,
                    "elemArr":  1,
                    "padding":  "nopad",
                    "folds": {
                      "dim_prop_func": [
                        {"Affine": {"alpha_": 16, "beta_": 0}},
                        {"Affine": {"alpha_": 0,  "beta_": 0}},
                        {"Affine": {"alpha_": 0,  "beta_": 0}},
                        {"Affine": {"alpha_": 1,  "beta_": 0}}
                      ],
                      "dim_prop_attr": [
                        {"factor_": 2,  "label_": "core_fold"},
                        {"factor_": 1,  "label_": "corelet_fold"},
                        {"factor_": 1,  "label_": "row_fold"},
                        {"factor_": 16, "label_": "elem_arr_0"}
                      ]
                    }
                  },
                  "out": {
                    "spatial":  3,
                    "temporal": 0,
                    "elemArr":  2,
                    "padding":  "nopad",
                    "folds": {
                      "dim_prop_func": [
                        {"Affine": {"alpha_": 128, "beta_": 0}},
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
                        {"factor_": 2,  "label_": "elem_arr_1"}
                      ]
                    }
                  }
                }
              }
            },
            {
              "nodeType_": "allocate",
              "name_": "output_alloc",
              "prev_": "input_alloc",
              "ldsIdx_": 1,
              "component_": "hbm",
              "layoutDimOrder_": ["mb", "out"],
              "maxDimSizes_": [32, 128],
              "startAddressCoreCorelet_": {
                "dim_prop_func": [{"Map": {}}, {"Const": {}}],
                "dim_prop_attr": [
                  {"factor_": 2, "label_": "core"},
                  {"factor_": 2, "label_": "corelet"}
                ],
                "data_": {
                  "[0, 0]": "16384",
                  "[1, 0]": "24576"
                }
              },
              "coordinates_": {
                "coordInfo": {
                  "mb": {
                    "spatial":  3,
                    "temporal": 0,
                    "elemArr":  1,
                    "padding":  "nopad",
                    "folds": {
                      "dim_prop_func": [
                        {"Affine": {"alpha_": 16, "beta_": 0}},
                        {"Affine": {"alpha_": 0,  "beta_": 0}},
                        {"Affine": {"alpha_": 0,  "beta_": 0}},
                        {"Affine": {"alpha_": 1,  "beta_": 0}}
                      ],
                      "dim_prop_attr": [
                        {"factor_": 2,  "label_": "core_fold"},
                        {"factor_": 1,  "label_": "corelet_fold"},
                        {"factor_": 1,  "label_": "row_fold"},
                        {"factor_": 16, "label_": "elem_arr_0"}
                      ]
                    }
                  },
                  "out": {
                    "spatial":  3,
                    "temporal": 0,
                    "elemArr":  2,
                    "padding":  "nopad",
                    "folds": {
                      "dim_prop_func": [
                        {"Affine": {"alpha_": 128, "beta_": 0}},
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
                        {"factor_": 2,  "label_": "elem_arr_1"}
                      ]
                    }
                  }
                }
              }
            }
          ],
          "labeledDs_": [
            {
              "ldsIdx_": 0,
              "dsName_": "gelu-Tensor0",
              "dsType_": "INPUT",
              "scale_": [1.0, 1.0],
              "wordLength": 2,
              "dataFormat_": "SEN169_FP16",
              "memOrg_": {
                "hbm": {"isPresent": 1},
                "lx": {"isPresent": 0}
              }
            },
            {
              "ldsIdx_": 1,
              "dsName_": "gelu-Tensor1",
              "dsType_": "OUTPUT",
              "scale_": [1.0, 1.0],
              "wordLength": 2,
              "dataFormat_": "SEN169_FP16",
              "memOrg_": {
                "hbm": {"isPresent": 1},
                "lx": {"isPresent": 0}
              }
            }
          ],
          "constantInfo_": "{}",
          "computeOp_": [
            {
              "exUnit": "sfp",
              "opFuncName": "gelufwd",
              "attributes_": {
                "dataFormat_": "SEN169_FP16",
                "fidelity_": "regular"
              },
              "location": "Inner",
              "inputLabeledDs": ["gelu-Tensor0-idx0"],
              "outputLabeledDs": ["gelu-Tensor1-idx1"]
            }
          ]
        }
      }
    ]
  }
}
```

---

| [← Previous: ConstantInfo](constantinfo.md) | [↑ Table of Contents](README.md) | [Next: Indirect Access Example →](indirect-access-example.md) |
|:--|:--:|--:|
