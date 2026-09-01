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
      "0": [[0, 0, 0, 0]],
      "1": [[0, 0, 0, 0]]
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
              }
            },
            {
              "nodeType_": "allocate",
              "name_": "output_alloc",
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
              }
            }
          ],
          "labeledDs_": [
            {
              "ldsIdx_": 0,
              "dsName_": "input",
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
              "dsName_": "output",
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
              "opFuncName": "gelu",
              "attributes_": {
                "dataFormat_": "SEN169_FP16",
                "fidelity_": "regular"
              },
              "location": "Inner",
              "inputLabeledDs": ["input"],
              "outputLabeledDs": ["output"]
            }
          ]
        }
      }
    ]
  }
}
```

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: Schema Validation →](validation_bkp.mp) |
|:--|:--:|--:|
