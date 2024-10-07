module {
  func.func @"torch-jit-export"(%arg0: !torch.vtensor<[?,?,16,64],f32>) -> !torch.vtensor<[?,?,1024],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "1.11.0"} {
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %0 = torch.vtensor.literal(dense<16> : tensor<si64>) : !torch.vtensor<[],si64>
    %1 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %2 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %3 = torch.vtensor.literal(dense<1024> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %4 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[?,?,16,64],f32> -> !torch.vtensor<[4],si64>
    %5 = torch.aten.index_select %4, %int0, %2 : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %6 = torch.aten.squeeze.dim %5, %int0 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[],si64>
    %7 = torch.aten._shape_as_tensor %arg0 : !torch.vtensor<[?,?,16,64],f32> -> !torch.vtensor<[4],si64>
    %8 = torch.aten.index_select %7, %int0, %1 : !torch.vtensor<[4],si64>, !torch.int, !torch.vtensor<[1],si64> -> !torch.vtensor<[1],si64>
    %9 = torch.aten.squeeze.dim %8, %int0 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[],si64>
    %10 = torch.aten.unsqueeze %6, %int0 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %11 = torch.aten.unsqueeze %9, %int0 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %12 = torch.prim.ListConstruct %10, %11, %3 : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.list<vtensor>
    %13 = torch.aten.cat %12, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[3],si64>
    %14 = torch.aten.slice.Tensor %13, %int0, %int0, %int1, %int1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %15 = torch.aten.item %14 : !torch.vtensor<[1],si64> -> !torch.int
    %16 = torch.aten.eq.int %15, %int0 : !torch.int, !torch.int -> !torch.bool
    %17 = torch.aten.Int.bool %16 : !torch.bool -> !torch.int
    %18 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?,16,64],f32>, !torch.int -> !torch.int
    %19 = torch.prim.NumToTensor.Scalar %17 : !torch.int -> !torch.vtensor<[],i1>
    %20 = torch.prim.NumToTensor.Scalar %18 : !torch.int -> !torch.vtensor<[],si64>
    %21 = torch.prim.NumToTensor.Scalar %15 : !torch.int -> !torch.vtensor<[],si64>
    %22 = torch.aten.where.self %19, %20, %21 : !torch.vtensor<[],i1>, !torch.vtensor<[],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %23 = torch.aten.item %22 : !torch.vtensor<[],si64> -> !torch.int
    %24 = torch.aten.slice.Tensor %13, %int0, %int1, %int2, %int1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %25 = torch.aten.item %24 : !torch.vtensor<[1],si64> -> !torch.int
    %26 = torch.aten.eq.int %25, %int0 : !torch.int, !torch.int -> !torch.bool
    %27 = torch.aten.Int.bool %26 : !torch.bool -> !torch.int
    %28 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[?,?,16,64],f32>, !torch.int -> !torch.int
    %29 = torch.prim.NumToTensor.Scalar %27 : !torch.int -> !torch.vtensor<[],i1>
    %30 = torch.prim.NumToTensor.Scalar %28 : !torch.int -> !torch.vtensor<[],si64>
    %31 = torch.prim.NumToTensor.Scalar %25 : !torch.int -> !torch.vtensor<[],si64>
    %32 = torch.aten.where.self %29, %30, %31 : !torch.vtensor<[],i1>, !torch.vtensor<[],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %33 = torch.aten.item %32 : !torch.vtensor<[],si64> -> !torch.int
    %34 = torch.aten.slice.Tensor %13, %int0, %int2, %int3, %int1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1],si64>
    %35 = torch.aten.item %34 : !torch.vtensor<[1],si64> -> !torch.int
    %36 = torch.aten.eq.int %35, %int0 : !torch.int, !torch.int -> !torch.bool
    %37 = torch.aten.Int.bool %36 : !torch.bool -> !torch.int
    %38 = torch.prim.NumToTensor.Scalar %37 : !torch.int -> !torch.vtensor<[],i1>
    %39 = torch.prim.NumToTensor.Scalar %35 : !torch.int -> !torch.vtensor<[],si64>
    %40 = torch.aten.where.self %38, %0, %39 : !torch.vtensor<[],i1>, !torch.vtensor<[],si64>, !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %41 = torch.aten.item %40 : !torch.vtensor<[],si64> -> !torch.int
    %42 = torch.prim.ListConstruct %23, %33, %41 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %43 = torch.aten.view %arg0, %42 : !torch.vtensor<[?,?,16,64],f32>, !torch.list<int> -> !torch.vtensor<[?,?,1024],f32>
    return %43 : !torch.vtensor<[?,?,1024],f32>
  }
}

