# ComputeOperation

Compute operation specification.

## Required Fields

- `exUnit`
- `opFuncName`
- `inputLabeledDs`
- `outputLabeledDs`

## Structure

```json
{
  "exUnit": "sfp" | "pt",
  "opFuncName": "<string>",
  "attributes_": <OperationAttributes>,
  "location": "Inner" | "Outer",
  "inputLabeledDs": [<string>, ...],
  "outputLabeledDs": [<string>, ...]
}
```

## Field Descriptions

### exUnit
- **Type**: `string`
- **Description**: Execution unit
- **Required**: Yes
- **Enum**: `["sfp", "pt"]`

### opFuncName
- **Type**: `string`
- **Description**: Operation function name — supports all torch-spyre generated operations
- **Required**: Yes

### attributes_
- **Type**: `OperationAttributes`
- **Description**: Operation attributes
- **Optional**: Yes

### location
- **Type**: `string`
- **Description**: Operation location
- **Optional**: Yes
- **Enum**: `["Inner", "Outer"]`

### inputLabeledDs
- **Type**: `array of string`
- **Description**: Input labeled data structure names
- **Required**: Yes

### outputLabeledDs
- **Type**: `array of string`
- **Description**: Output labeled data structure names
- **Required**: Yes

## Example

```json
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
```

## Supported Operations

The following table lists operations that torch-spyre can generate in SDSC bundles:

| Category | OpFunc String | Constants | Description |
|----------|---------------|-----------|-------------|
| **Matmul** | `"batchmatmul"` | — | Batch matrix multiplication (DL16) |
| | `"batchmatmulfp8"` | — | Batch matmul (FP8) |
| | `"batchmatmulint4"` | — | Batch matmul (INT4) |
| | `"batchmatmulint8"` | — | Batch matmul (INT8) |
| **Convolution** | `"conv2d"` | — | 2D convolution (DL16) |
| | `"conv2dfp8"` | — | 2D convolution (FP8) |
| | `"conv2dint4"` | — | 2D convolution (INT4) |
| | `"conv2dint8"` | — | 2D convolution (INT8) |
| **Broadcast** | `"add"` | — | Element-wise addition (supports broadcast on any dimensions) |
| | `"batchnormfwd"` | — | Batch normalization forward |
| | `"biasadd"` | — | Bias addition |
| | `"equal"` | — | Element-wise equality comparison |
| | `"fnms"` | — | Fused negative multiply-subtract |
| | `"greaterequal"` | — | Element-wise greater than or equal |
| | `"layernormnorm"` | — | Layer normalization (norm step) |
| | `"lesserequal"` | — | Element-wise less than or equal |
| | `"maximum"` | — | Element-wise maximum |
| | `"minimum"` | — | Element-wise minimum |
| | `"mul"` | — | Element-wise multiplication |
| | `"notequal"` | — | Element-wise not equal comparison |
| | `"realdiv"` | — | Element-wise division |
| | `"revsub"` | — | Reverse subtraction (b - a) |
| | `"sub"` | — | Element-wise subtraction |
| | `"where3"` | — | Conditional selection (ternary where) |
| **Unary** | `"abs"` | — | Absolute value |
| | `"clip"` | `clipMin`, `clipMax` | Clipping operation |
| | `"exp"` | — | Exponential |
| | `"fastexp"` | — | Fast exponential approximation |
| | `"fastsigmoid"` | — | Fast sigmoid approximation |
| | `"gelufwd"` | — | GELU activation |
| | `"identity"` | — | Identity operation (pass-through) |
| | `"layernormscale"` | `eps` | Layer normalization (scale step) |
| | `"leakyrelufwd"` | — | Leaky ReLU activation |
| | `"log"` | — | Natural logarithm |
| | `"mish"` | — | Mish activation |
| | `"neg"` | — | Negation |
| | `"reciprocal"` | — | Reciprocal (1/x) |
| | `"relufwd"` | — | ReLU activation |
| | `"relu6fwd"` | — | ReLU6 activation |
| | `"rsqrt"` | — | Reciprocal square root |
| | `"sigmoid"` | — | Sigmoid activation |
| | `"silu"` | — | SiLU/Swish activation |
| | `"softplus"` | `softplusBeta`, `softplusThresh` | Softplus activation |
| | `"sqrt"` | — | Square root |
| | `"tanh"` | — | Hyperbolic tangent |
| **Reduction** | `"absmax"` | — | Absolute max reduction (stick) |
| | `"absmaxnonstick"` | — | Absolute max reduction (non-stick) |
| | `"exx2"` | `exx2scale` | E[x²] calculation |
| | `"exx2_zeromean"` | — | E[x²] with zero mean |
| | `"max"` | — | Max reduction (stick) |
| | `"maxnonstick"` | — | Max reduction (non-stick) |
| | `"mean"` | `scaling_factor` | Mean reduction (stick) |
| | `"meannonstick"` | `scaling_factor` | Mean reduction (non-stick) |
| | `"min"` | — | Min reduction (stick) |
| | `"minnonstick"` | — | Min reduction (non-stick) |
| | `"quantscalepertoken"` | — | Quantization scale per token |
| | `"quantscalepertokenfp8"` | `clipMin`, `clipMax`, `mulConst` | FP8 quantization scale per token |
| | `"sum"` | — | Sum reduction (stick) |
| | `"sumnonstick"` | — | Sum reduction (non-stick) |
| **Pooling** | `"avgpoolfwd"` | `nmap` | Average pooling (nmap = 1/(kh×kw)) |
| | `"avgpoolnmapfwd"` | — | Average pooling with nmap |
| | `"depthwiseconv2dnative"` | — | Depthwise convolution |
| | `"maxpoolfwd"` | — | Max pooling |
| **Scan** | `"maskbyindex"` | — | Mask by index operation |
| | `"topkindex"` | — | Top-K indices |
| | `"topkvalue"` | — | Top-K values |
| **Quantization** | `"csqint4"` | `scaleact`, `shiftact` | Scale/shift DL16 to INT4 (ch packing) |
| | `"csqint4wt"` | `scaleact`, `shiftact` | Scale/shift DL16 to INT4 (wt packing) |
| | `"csqint8ch"` | `scaleact`, `shiftact` | Scale/shift DL16 to INT8 (ch packing) |
| | `"csqint8mb"` | `scaleact`, `shiftact` | Scale/shift DL16 to INT8 (mb packing) |
| | `"csqint8wt"` | `scaleact`, `shiftact` | Scale/shift DL16 to INT8 (wt packing) |
| | `"dl16tofp32"` | — | Convert DL16 to FP32 |
| | `"fp32todl16"` | — | Quantize FP32 to DL16 |
| | `"fp8todl16"` | — | Convert FP8<1,4,3> to DL16 |
| | `"qfp8ch"` | — | Quantize DL16 to FP8<1,4,3> (ch packing) |
| | `"qfp8mb"` | — | Quantize DL16 to FP8<1,4,3> (mb packing) |
| | `"qfp8wt"` | — | Quantize DL16 to FP8<1,4,3> (wt packing) |

**Note**: The `opFuncName` field accepts any operation name that torch-spyre generates. The table above lists common operations, but torch-spyre may generate additional operations based on the PyTorch model being compiled.

---

[← Back to Table of Contents](README.md)
