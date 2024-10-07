module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @"torch-jit-export"(%arg0: tensor<?x?x16x64xf32>) -> tensor<?x?x1024xf32> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024_i64 = arith.constant 1024 : i64
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x16x64xf32>
    %0 = arith.index_cast %dim : index to i64
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x16x64xf32>
    %1 = arith.index_cast %dim_0 : index to i64
    %2 = arith.cmpi slt, %0, %c0_i64 : i64
    %3 = arith.select %2, %c1_i64, %0 : i64
    %4 = arith.extui %2 : i1 to i64
    %5 = arith.muli %3, %1 : i64
    %6 = arith.addi %4, %c1_i64 : i64
    %7 = arith.cmpi slt, %1, %c0_i64 : i64
    %8 = arith.select %7, %3, %5 : i64
    %9 = arith.select %7, %6, %4 : i64
    %10 = arith.addi %9, %c1_i64 : i64
    %11 = arith.cmpi sle, %10, %c1_i64 : i64
    cf.assert %11, "must have at most one inferred (negative) dimension"
    %12 = arith.muli %0, %1 : i64
    %13 = arith.muli %12, %c1024_i64 : i64
    %14 = arith.divsi %13, %8 : i64
    %15 = arith.select %2, %14, %0 : i64
    %16 = arith.select %7, %14, %1 : i64
    %from_elements = tensor.from_elements %15, %16, %14 : tensor<3xi64>
    %reshape = tensor.reshape %arg0(%from_elements) : (tensor<?x?x16x64xf32>, tensor<3xi64>) -> tensor<?x?x1024xf32>
    return %reshape : tensor<?x?x1024xf32>
  }
}

