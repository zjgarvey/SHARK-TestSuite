#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @addgraph(%arg0: tensor<f32>, %arg1: tensor<6xf32>) -> tensor<6xf32> {
    %0 = tensor.empty() : tensor<6xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<f32>, tensor<6xf32>) outs(%0 : tensor<6xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<6xf32>
    return %1 : tensor<6xf32>
  }
}

