module {
  func.func @test_eyelike_with_dtype(%arg0: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f64> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.EyeLike"(%arg0) {torch.onnx.dtype = 11 : si64} : (!torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],f64> 
    return %0 : !torch.vtensor<[3,4],f64>
  }
}

