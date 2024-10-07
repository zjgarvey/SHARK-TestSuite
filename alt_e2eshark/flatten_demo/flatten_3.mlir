#map = affine_map<() -> ()>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @"torch-jit-export"(%arg0: tensor<?x?x16x64xf32>) -> tensor<?x?x1024xf32> {
    %cst = arith.constant dense<1024> : tensor<1xi64>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024_i64 = arith.constant 1024 : i64
    %cst_0 = arith.constant dense<16> : tensor<i64>
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x16x64xf32>
    %0 = arith.index_cast %dim : index to i64
    %1 = tensor.empty() : tensor<1xi64>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x16x64xf32>
    %2 = arith.index_cast %dim_1 : index to i64
    %3 = linalg.fill ins(%0 : i64) outs(%1 : tensor<1xi64>) -> tensor<1xi64>
    %4 = linalg.fill ins(%2 : i64) outs(%1 : tensor<1xi64>) -> tensor<1xi64>
    %concat = tensor.concat dim(0) %3, %4, %cst : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %extracted_slice = tensor.extract_slice %concat[0] [1] [1] : tensor<3xi64> to tensor<1xi64>
    %extracted = tensor.extract %extracted_slice[%c0] : tensor<1xi64>
    %5 = arith.cmpi eq, %extracted, %c0_i64 : i64
    %6 = tensor.empty() : tensor<i1>
    %7 = linalg.fill ins(%5 : i1) outs(%6 : tensor<i1>) -> tensor<i1>
    %8 = tensor.empty() : tensor<i64>
    %9 = linalg.fill ins(%0 : i64) outs(%8 : tensor<i64>) -> tensor<i64>
    %10 = linalg.fill ins(%extracted : i64) outs(%8 : tensor<i64>) -> tensor<i64>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%7, %9, %10 : tensor<i1>, tensor<i64>, tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i1, %in_9: i64, %in_10: i64, %out: i64):
      %41 = arith.select %in, %in_9, %in_10 : i64
      linalg.yield %41 : i64
    } -> tensor<i64>
    %extracted_2 = tensor.extract %11[] : tensor<i64>
    %extracted_slice_3 = tensor.extract_slice %concat[1] [1] [1] : tensor<3xi64> to tensor<1xi64>
    %extracted_4 = tensor.extract %extracted_slice_3[%c0] : tensor<1xi64>
    %12 = arith.cmpi eq, %extracted_4, %c0_i64 : i64
    %13 = linalg.fill ins(%12 : i1) outs(%6 : tensor<i1>) -> tensor<i1>
    %14 = linalg.fill ins(%2 : i64) outs(%8 : tensor<i64>) -> tensor<i64>
    %15 = linalg.fill ins(%extracted_4 : i64) outs(%8 : tensor<i64>) -> tensor<i64>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%13, %14, %15 : tensor<i1>, tensor<i64>, tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i1, %in_9: i64, %in_10: i64, %out: i64):
      %41 = arith.select %in, %in_9, %in_10 : i64
      linalg.yield %41 : i64
    } -> tensor<i64>
    %extracted_5 = tensor.extract %16[] : tensor<i64>
    %extracted_slice_6 = tensor.extract_slice %concat[2] [1] [1] : tensor<3xi64> to tensor<1xi64>
    %extracted_7 = tensor.extract %extracted_slice_6[%c0] : tensor<1xi64>
    %17 = arith.cmpi eq, %extracted_7, %c0_i64 : i64
    %18 = linalg.fill ins(%17 : i1) outs(%6 : tensor<i1>) -> tensor<i1>
    %19 = linalg.fill ins(%extracted_7 : i64) outs(%8 : tensor<i64>) -> tensor<i64>
    %20 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%18, %cst_0, %19 : tensor<i1>, tensor<i64>, tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i1, %in_9: i64, %in_10: i64, %out: i64):
      %41 = arith.select %in, %in_9, %in_10 : i64
      linalg.yield %41 : i64
    } -> tensor<i64>
    %extracted_8 = tensor.extract %20[] : tensor<i64>
    %21 = arith.cmpi slt, %extracted_2, %c0_i64 : i64
    %22 = arith.select %21, %c1_i64, %extracted_2 : i64
    %23 = arith.extui %21 : i1 to i64
    %24 = arith.muli %22, %extracted_5 : i64
    %25 = arith.addi %23, %c1_i64 : i64
    %26 = arith.cmpi slt, %extracted_5, %c0_i64 : i64
    %27 = arith.select %26, %22, %24 : i64
    %28 = arith.select %26, %25, %23 : i64
    %29 = arith.muli %27, %extracted_8 : i64
    %30 = arith.addi %28, %c1_i64 : i64
    %31 = arith.cmpi slt, %extracted_8, %c0_i64 : i64
    %32 = arith.select %31, %27, %29 : i64
    %33 = arith.select %31, %30, %28 : i64
    %34 = arith.cmpi sle, %33, %c1_i64 : i64
    cf.assert %34, "must have at most one inferred (negative) dimension"
    %35 = arith.muli %0, %2 : i64
    %36 = arith.muli %35, %c1024_i64 : i64
    %37 = arith.divsi %36, %32 : i64
    %38 = arith.select %21, %37, %extracted_2 : i64
    %39 = arith.select %26, %37, %extracted_5 : i64
    %40 = arith.select %31, %37, %extracted_8 : i64
    %from_elements = tensor.from_elements %38, %39, %40 : tensor<3xi64>
    %reshape = tensor.reshape %arg0(%from_elements) : (tensor<?x?x16x64xf32>, tensor<3xi64>) -> tensor<?x?x1024xf32>
    return %reshape : tensor<?x?x1024xf32>
  }
}

