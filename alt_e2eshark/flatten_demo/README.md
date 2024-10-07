This IR was extracted from the test:

migraphx_ORT__bert_large_uncased_1

- `flatten_0.mlir` is taken from that test directly
- `flatten_1.mlir` is the result of `-convert-torch-onnx-to-torch` on prev
- `flatten_2.mlir` is the result of `-torch-lower-to-backend-contract` on prev
- `flatten_3.mlir` is the result of `-torch-backend-to-linalg-on-tensors-backend-pipeline` on prev

- `good_flatten_2.mlir` is what I'd like `flatten_2.mlir` to be.
- Hilariously, `ruined_good_flatten_2.mlir` is the result of `-torch-lower-to-backend-contract` when applied to `good_flatten_2.mlir`
- `good_flatten_3.mlir` is the result of `-torch-backend-to-linalg-on-tensors-backend-pipeline` on `good_flatten_2.mlir`
- `ruined_good_flatten_3.mlir` is the result of `-torch-backend-to-linalg-on-tensors-backend-pipeline` on `ruined_good_flatten_2.mlir`
