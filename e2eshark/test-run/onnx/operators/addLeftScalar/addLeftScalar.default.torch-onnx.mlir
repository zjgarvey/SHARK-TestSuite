module {
  func.func @addgraph(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<[],f32>, !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],f32> 
    return %0 : !torch.vtensor<[6],f32>
  }
}

