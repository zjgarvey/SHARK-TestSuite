module {
  func.func @"torch-jit-export"(%683 : !torch.vtensor<[?,?,16,64],f32>) -> (!torch.vtensor<[?,?,1024],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "1.11.0"} {
    // IR grabbed from migraphx_ORT__bert_large_uncased_1
    %684 = torch.operator "onnx.Shape"(%683) : (!torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[4],si64> 
    // zero
    %685 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__21> : tensor<si64>} : () -> !torch.vtensor<[],si64> 
    // dim zero of %683
    %686 = torch.operator "onnx.Gather"(%684, %685) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[4],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64> 
    %687 = torch.operator "onnx.Shape"(%683) : (!torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[4],si64> 
    // one
    %688 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__22> : tensor<si64>} : () -> !torch.vtensor<[],si64> 
    // dim one of %683
    %689 = torch.operator "onnx.Gather"(%687, %688) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[4],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],si64> 
    %690 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %691 = torch.operator "onnx.Unsqueeze"(%686, %690) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
    %692 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %693 = torch.operator "onnx.Unsqueeze"(%689, %692) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
    // [dim0, dim1, 1024]
    %261 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_onnx__Concat_3209> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %694 = torch.operator "onnx.Concat"(%691, %693, %261) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3],si64> 
    // this reshape is a flatten op
    %695 = torch.operator "onnx.Reshape"(%683, %694) : (!torch.vtensor<[?,?,16,64],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[?,?,1024],f32> 
    return %695 : !torch.vtensor<[?,?,1024],f32>
    }
}

{-#
  dialect_resources: {
    builtin: {
      __21: "0x080000000000000000000000",
      __22: "0x080000000100000000000000",
      _onnx__Concat_3209: "0x080000000004000000000000"
    }
  }
#-}

