module {
  func.func @"torch-jit-export"(%arg0: !torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[?,?,1024],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "1.11.0"} {
    // This is the simplest equivalent Torch IR
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.aten.flatten.using_ints %arg0, %int2, %int3 : !torch.vtensor<[?,?,16,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?,1024],f32>
    return %0 : !torch.vtensor<[?,?,1024],f32>
  }
}

