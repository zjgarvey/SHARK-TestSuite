module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @"torch-jit-export"(%arg0: tensor<?x?x16x64xf32>) -> tensor<?x?x1024xf32> {
    // Amazing. This is the dream.
    %collapsed = tensor.collapse_shape %arg0 [[0], [1], [2, 3]] : tensor<?x?x16x64xf32> into tensor<?x?x1024xf32>
    return %collapsed : tensor<?x?x1024xf32>
  }
}

