# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor

from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_with_name

@register_with_name("transpose_0231")
class TransposeModelInfo(OnnxModelInfo):
    def construct_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [1, 21, 513, 513])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [1, 513, 513, 21])
        node_list = []
        app_node = lambda op_ty, inputs, outputs, **kwargs: node_list.append(
            make_node(op_ty, inputs, outputs, **kwargs)
        )
        app_node("Transpose", ["X"], ["Y"], perm = [0, 2, 3, 1]) 
        graph = make_graph(node_list, "main", [X], [Y])
        model = make_model(graph)
        onnx.save(model, self.model)