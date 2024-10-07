module {
  func.func @"torch-jit-export"(%arg0: !torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[?,?,1024],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "1.11.0"} {
    %int-1 = torch.constant.int -1
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?,16,64],f32>, !torch.int -> !torch.int
    %1 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?,16,64],f32>, !torch.int -> !torch.int
    %2 = torch.prim.ListConstruct %0, %1, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %arg0, %2 : !torch.vtensor<[?,?,16,64],f32>, !torch.list<int> -> !torch.vtensor<[?,?,1024],f32>
    return %3 : !torch.vtensor<[?,?,1024],f32>
  }
}

