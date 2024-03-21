This 

```JSON
  node {
    input: "/decoder/embed_tokens/Gather_output_0"
    input: "/decoder/embed_positions/Gather_output_0"
    output: "/decoder/Add_3_output_0"
    name: "/decoder/Add_3"
    op_type: "Add"
  }
```

converts to this

```MLIR
%1134 = torch.operator "onnx.Add"(%1119, %1129) : (!torch.vtensor<[],f32>, !torch.vtensor<[?,?,768],f32>) -> !torch.vtensor<[],f32> 
```

The arguments:

```JSON
  node {
    input: "model.decoder.embed_tokens.weight"
    input: "/decoder/Reshape_output_0"
    output: "/decoder/embed_tokens/Gather_output_0"
    name: "/decoder/embed_tokens/Gather"
    op_type: "Gather"
  }
  node {
    input: "model.decoder.embed_positions.weight"
    input: "/decoder/embed_positions/Add_output_0"
    output: "/decoder/embed_positions/Gather_output_0"
    name: "/decoder/embed_positions/Gather"
    op_type: "Gather"
  }
```

converted (respectively)

```MLIR
%1119 = torch.operator "onnx.Gather"(%0, %1113) : (!torch.vtensor<[50272,768],f32>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> 
%1129 = torch.operator "onnx.Gather"(%1, %1124) : (!torch.vtensor<[2050,768],f32>, !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?,768],f32> 
```

Their arguments

```JSON
  node {
    input: "input_ids"
    input: "/decoder/Concat_output_0"
    output: "/decoder/Reshape_output_0"
    name: "/decoder/Reshape"
    op_type: "Reshape"
  }
  node {
    input: "/decoder/embed_positions/Slice_output_0"
    input: "/decoder/embed_positions/Constant_6_output_0"
    output: "/decoder/embed_positions/Add_output_0"
    name: "/decoder/embed_positions/Add"
    op_type: "Add"
  }
```

converted respectively

```MLIR
%0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_model.decoder.embed_tokens.weight> : tensor<50272x768xf32>} : () -> !torch.vtensor<[50272,768],f32> 
%1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_model.decoder.embed_positions.weight> : tensor<2050x768xf32>} : () -> !torch.vtensor<[2050,768],f32> 
%1113 = torch.operator "onnx.Reshape"(%arg0, %1107) : (!torch.vtensor<[?,?],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[],si64> 
%1124 = torch.operator "onnx.Add"(%1118, %392) : (!torch.vtensor<[?,?],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[?,?],si64> 
```

It seems that the reshape is an issue, no?

Here is the non input argument for the reshape:

```JSON
  node {
    input: "/decoder/Constant_2_output_0"
    input: "/decoder/Unsqueeze_output_0"
    output: "/decoder/Concat_output_0"
    name: "/decoder/Concat"
    op_type: "Concat"
    attribute {
      name: "axis"
      type: INT
      i: 0
    }
  }
```

which converts to

```MLIR
%1107 = torch.operator "onnx.Concat"(%345, %1097) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2],si64> 
```

-------------------------------------------

If you meant literally adjacent:

```JSON
  node {
    input: "model.decoder.embed_positions.weight"
    input: "/decoder/embed_positions/Add_output_0"
    output: "/decoder/embed_positions/Gather_output_0"
    name: "/decoder/embed_positions/Gather"
    op_type: "Gather"
  }
  node {
    input: "/decoder/Slice_output_0"
    input: "/decoder/Constant_7_output_0"
    output: "/decoder/Squeeze_output_0"
    name: "/decoder/Squeeze"
    op_type: "Squeeze"
  }
  node {
    input: "/decoder/Slice_1_output_0"
    input: "/decoder/Constant_14_output_0"
    output: "/decoder/Squeeze_1_output_0"
    name: "/decoder/Squeeze_1"
    op_type: "Squeeze"
  }
  node {
    input: "/decoder/ConstantOfShape_3_output_0"
    input: "/decoder/Constant_34_output_0"
    output: "/decoder/Mul_1_output_0"
    name: "/decoder/Mul_1"
    op_type: "Mul"
  }
  node {
    input: "/decoder/Shape_5_output_0"
    output: "/decoder/ConstantOfShape_2_output_0"
    name: "/decoder/ConstantOfShape_2"
    op_type: "ConstantOfShape"
    attribute {
      name: "value"
      type: TENSOR
      t {
        dims: 1
        data_type: 7
        raw_data: "\001\000\000\000\000\000\000\000"
      }
    }
  }
  node {
    input: "/decoder/embed_tokens/Gather_output_0"
    input: "/decoder/embed_positions/Gather_output_0"
    output: "/decoder/Add_3_output_0"
    name: "/decoder/Add_3"
    op_type: "Add"
  }
  node {
    input: "/decoder/Squeeze_output_0"
    output: "/decoder/Cast_output_0"
    name: "/decoder/Cast"
    op_type: "Cast"
    attribute {
      name: "to"
      type: INT
      i: 7
    }
  }
  node {
    input: "/decoder/Squeeze_1_output_0"
    input: "onnx::Unsqueeze_266"
    output: "/decoder/Unsqueeze_3_output_0"
    name: "/decoder/Unsqueeze_3"
    op_type: "Unsqueeze"
  }
  node {
    input: "/decoder/Reshape_3_output_0"
    input: "/decoder/Mul_1_output_0"
    output: "/decoder/Equal_1_output_0"
    name: "/decoder/Equal_1"
    op_type: "Equal"
  }
  node {
    input: "/decoder/ConstantOfShape_2_output_0"
    input: "/decoder/Constant_24_output_0"
    output: "/decoder/Mul_output_0"
    name: "/decoder/Mul"
    op_type: "Mul"
  }
  node {
    input: "/decoder/Add_3_output_0"
    output: "/decoder/layers.0/self_attn_layer_norm/ReduceMean_output_0"
    name: "/decoder/layers.0/self_attn_layer_norm/ReduceMean"
    op_type: "ReduceMean"
    attribute {
      name: "axes"
      type: INTS
      ints: -1
    }
  }
```


```MLIR
    %1129 = torch.operator "onnx.Gather"(%1, %1124) : (!torch.vtensor<[2050,768],f32>, !torch.vtensor<[?,?],si64>) -> !torch.vtensor<[?,?,768],f32> 
    %1130 = torch.operator "onnx.Squeeze"(%1125, %353) : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],si64> 
    %1131 = torch.operator "onnx.Squeeze"(%1126, %360) : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],si64> 
    %1132 = torch.operator "onnx.Mul"(%1127, %383) : (!torch.vtensor<[?],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[?],si64> 
    %1133 = torch.operator "onnx.ConstantOfShape"(%1128) {torch.onnx.value = dense_resource<__739> : tensor<1xsi64>} : (!torch.vtensor<[1],si64>) -> !torch.vtensor<[?],si64> 
    %1134 = torch.operator "onnx.Add"(%1119, %1129) : (!torch.vtensor<[],f32>, !torch.vtensor<[?,?,768],f32>) -> !torch.vtensor<[],f32> 
    %1135 = torch.operator "onnx.Cast"(%1130) {torch.onnx.to = 7 : si64} : (!torch.vtensor<[],si64>) -> !torch.vtensor<[],si64> 
    %1136 = torch.operator "onnx.Unsqueeze"(%1131, %361) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
    %1137 = torch.operator "onnx.Equal"(%1115, %1132) : (!torch.vtensor<[4],si64>, !torch.vtensor<[?],si64>) -> !torch.vtensor<[4],i1> 
    %1138 = torch.operator "onnx.Mul"(%1133, %373) : (!torch.vtensor<[?],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[?],si64> 
    %1139 = torch.operator "onnx.ReduceMean"(%1134) {torch.onnx.axes = [-1 : si64]} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
 ```