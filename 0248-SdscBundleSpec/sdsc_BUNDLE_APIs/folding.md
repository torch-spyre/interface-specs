# Folding

The main folding related data structures are as follows:

```
class FoldManger {
    FoldFunction<Dtype>* parent_func_ = nullptr;
    fm_dim_prop dim_prop_;
}

using fm_dim_prop = std::vector<std::pair<const FoldDimProp*, BaseFuncType>>;

class FoldDimProp {
    unit32_t factor_;
    std::string label_;
}

enum class BaseFuncType {
    Constant = 0,
    Map = 1,
    Affine = 2,
    WkSplit = 3,
    Unknown = 4
};
```

In the SDSC, these structures are transformed to json of following format:

```
"dim_prop_func": [
    {
        "Affine": {"alpha_":1,"beta_":0}
    }
],

"dim_prop_attr": [
    {"factor_":1,"label_":"time"}
],

"data_": {"[0]":"0"}
```

The above generic format is used for a few fields such as `sdsc.sdscFolds_`, `scheduleTree.startAddressCoreCorelet`,
`scheduleTree.coordinates_.coordInfo.<dimname>.folds`. The usage of this generic structure
for different cases is described below.

## scheduleTree.startAddressCoreCorelet_

This field lists the starting memory addresses for each core, corelet, and timesteps for each of the tensors used.
Hence, it is part of allocate node of schedule tree. In this case, the `dim_prop_func` field is almost always set as follows.

```
"dim_prop_func": [
    { "Map": {} },
    {  "Const": {} },
]
```

```
"dim_prop_attr": [
    {"factor_":20,"label_":"core"},
    {"factor_":1,"label_":"corelet"},
    {"factor_":1,"label_":"time"}
],
```

Now, the `Map{}` entry in the first structure indicates that the start addresses for the tensor for various cores are
provided in the `"data_"` map. E.g.:

```
"data_": {
    "[0, 0, 0]":"0","[1, 0, 0]":"0","[2, 0, 0]":"0","[3, 0, 0]":"0","[4, 0, 0]":"0","[5, 0, 0]":"128",
    "[6, 0, 0]":"128","[7, 0, 0]":"128","[8, 0, 0]":"128","[9, 0, 0]":"128","[10, 0, 0]":"256",
    "[11, 0, 0]":"256","[12, 0, 0]":"256","[13, 0, 0]":"256","[14, 0, 0]":"256","[15, 0, 0]":"384",
    "[16, 0, 0]":"384","[17, 0, 0]":"384","[18, 0, 0]":"384","[19, 0, 0]":"384"
}
```

Rules for determining the addresses depend on how a tensor's dimensions are split.

Start addresses in **data_** field can be symbolic. In such a case, instead of real addresses, symbolic ids
are specified as values in the `data_` map. Actual values for the symbols are provided in the bundle mlir file that invokes
the sdsc. Refer to the [MLIR Bundle API](02-mlir-bundle-api.md).

The above example corresponds to a case when the first tensor has a 4-way split. Each slice is used by 5 cores
and hence the start address remains the same for 4 of the 20 cores. Each slice is just 128 bytes long, and hence
the start addresses of successive slices are 128 bytes apart.

## scheduleTree allocate Node's coordinates_.coordInfo.\<dim name\>.folds

Under scheduleTree, the `dim_prop_func` and `dim_prop_attr` sub-structures are used to describe how each dimension
of a tensor is progressively split across cores, corelets, rows, and the final leaf entities, referred to using
`dim_prop_attr` sub-structure labels `core_fold`, `corelet_fold`, `row_fold`, `elem_arr_0`, and `elem_arr_1`, respectively.
Consider a tensor with dimensions x, in, and out with sizes given by [4, 2880, 2880]. The x dimension has a
4-way split and the out dimension, a 5-way split while the in dimension is not split. The total number of splits
(tensor sub-blocks) is hence 20, with each split assigned to a core, for 20 cores in all. scheduleTree's
`coordinates_` provides details of these splits.

```
"coordinates_": {
    "coordInfo": {
        <dim_name>: {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 1 or 2,
            "padding": <padding type>,
            "folds": {
                "dim_prop_func": [
                    { "Affine": {"alpha_":<number of cores spanned by each core-wise split>, "beta_":0} },
                    { "Affine": {"alpha_":1, "beta_":0} },
                    { "Affine": {"alpha_":1, "beta_":0} },
                    { "Affine": {"alpha_":number of lower-most splits, typically 1, "beta_":0} },
                ],

                "dim_prop_attr": [
                    {"factor_": <number of core-wise splits >, "label_":"core_fold"},
                    {"factor_":<number of corelet splits>, "label_":"corelet_fold"},
                    {"factor_":<number of row splits>, "label_":"row_fold"},
                    {"factor_":<number of elements in the lower-most slice>, "label_":"elem_arr_0"}
                ]
            }
        }
    }
}
```

`coordInfo.spatial` is typically 3, indicating that there are 3 spatial splits, along cores, corelets, and rows.

`coordInfo.temporal` is set to 0 from the front end.

`coordInfo.elemArr` is 1 for non-stick dimensions and 2 for stick dimensions. It is 2 for stick dimensions as the elements
along the stick dimension are broken down into sticks. While a stick is at the lowest level, the number of sticks in a
dimension's slice is at the next higher level.

For a stick dimension, `dim_prop_attr` includes an entry with label set to `elem_arr_1`, whose corresponding `factor`
field would denote the number of sticks in each slice corresponding to a core, corelet, row combination, that is, the slice size/# of elements per stick. `factor_` corresponding to `elem_arr_0` would indicate the number of elements per stick.

For our example tensor, folds field for various dimensions would be as follows:

### For dimension **x**:

```
"folds": {
    "dim_prop_func": [
        { "Affine": {"alpha_":1,"beta_":0} },
        { "Affine": {"alpha_":0,"beta_":0} },
        { "Affine": {"alpha_":0,"beta_":0} },
        { "Affine": {"alpha_":1,"beta_":0} }
    ],

    "dim_prop_attr": [
        {"factor_":4,"label_":"core_fold"},
        {"factor_":1,"label_":"corelet_fold"},
        {"factor_":1,"label_":"row_fold"},
        {"factor_":1,"label_":"elem_arr_0"}
    ]
}
```

### For dimension **out**:

```
"folds": {
    "dim_prop_func": [
        { "Affine": {"alpha_":1,"beta_":0} },
        { "Affine": {"alpha_":0,"beta_":0} },
        { "Affine": {"alpha_":0,"beta_":0} },
        { "Affine": {"alpha_":1,"beta_":0} }
    ],

    "dim_prop_attr": [
        {"factor_":4,"label_":"core_fold"},
        {"factor_":1,"label_":"corelet_fold"},
        {"factor_":1,"label_":"row_fold"},
        {"factor_":1,"label_":"elem_arr_0"}
    ]
}
```

Since out dimension is a stick dimension, it also includes details for `elem_arr_1`.

### For dimension **in**:

```
"folds": {
    "dim_prop_func": [
        { "Affine": {"alpha_":2880,"beta_":0} },
        { "Affine": {"alpha_":0,"beta_":0} },
        { "Affine": {"alpha_":0,"beta_":0} },
        { "Affine": {"alpha_":1,"beta_":0} }
    ],

    "dim_prop_attr": [
        {"factor_":1,"label_":"core_fold"},
        {"factor_":1,"label_":"corelet_fold"},
        {"factor_":1,"label_":"row_fold"},
        {"factor_":2880,"label_":"elem_arr_0"}
    ]
}
```

---

| [← Back to Table of Contents](README.md) | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | [Next: FoldProperty →](foldproperty.md) |
|:--|:--:|--:|
