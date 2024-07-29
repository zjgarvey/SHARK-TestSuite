# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
import onnx
import torch
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_tensor,
)

from e2e_testing.framework import OnnxModelInfo
from e2e_testing.storage import TestTensors
from e2e_testing.registry import register_test, register_with_name

class ResizeTransposeQModel(OnnxModelInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.construct_model()

    def construct_nodes(self):
        node_list = []
        app_node = lambda op_ty, inputs, outputs, **kwargs: node_list.append(
            make_node(op_ty, inputs, outputs, **kwargs)
        )

        ST = make_tensor("ST",TensorProto.FLOAT, [], [0.25])
        ZPT = make_tensor("ZPT",TensorProto.INT8, [], [0])
        C0T = make_tensor("C0T",TensorProto.FLOAT, [4], [1.0, 1.0, 7.89230776, 7.89230776])
        C1T = make_tensor("C1T",TensorProto.FLOAT, [8], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        app_node("Constant",[],["S"],value=ST)
        app_node("Constant",[],["ZP"],value=ZPT)
        app_node("Constant",[],["C0"],value=C0T)
        app_node("Constant",[],["C1"],value=C1T)

        app_node("QuantizeLinear",["X0","S","ZP"], ["QX0"])
        app_node("DequantizeLinear", ["QX0","S","ZP"], ["DQX0"])
        app_node("Resize",["DQX0","C1","C0"],["X1"], mode="linear")
        app_node("QuantizeLinear",["X1","S","ZP"], ["QX1"])
        app_node("DequantizeLinear", ["QX1","S","ZP"], ["DQX1"])
        app_node("Transpose", ["DQX1"], ["X2"], perm = [0, 2, 3, 1]) 
        app_node("QuantizeLinear",["X2","S","ZP"], ["QX2"])
        app_node("DequantizeLinear", ["QX2","S","ZP"], ["DQX2"])
        return node_list
    
    def construct_io_value_info(self):
        input_vi = [make_tensor_value_info("X0", TensorProto.FLOAT, [1, 21, 65, 65])]
        output_vi = [make_tensor_value_info("DQX2", TensorProto.FLOAT, [1, 513, 513, 21])]
        return input_vi, output_vi

    def construct_model(self):
        node_list = self.construct_nodes()
        input, output = self.construct_io_value_info()
        graph = make_graph(node_list, "main", input, output)
        model = make_model(graph)
        onnx.save(model, self.model)

# register_test(ResizeTransposeQModel, "resize_tq")

class AnotherOne(ResizeTransposeQModel):
    def construct_inputs(self):
        # input = torch.Tensor([[[[0.42, 0.93], [0.27, 0.06]]]]).to(dtype=torch.float32)
        # this is the quantized result:
        input = torch.Tensor([[[[0.5, 1.0], [0.25, 0.00]]]]).to(dtype=torch.float32) 
        return TestTensors((input,))

    def construct_nodes(self):
        node_list = []
        app_node = lambda op_ty, inputs, outputs, **kwargs: node_list.append(
            make_node(op_ty, inputs, outputs, **kwargs)
        )
        ST = make_tensor("ST",TensorProto.FLOAT, [], [0.25])
        ZPT = make_tensor("ZPT",TensorProto.INT8, [], [0])
        C0T = make_tensor("C0T",TensorProto.FLOAT, [4], [1.0, 1.0, 1.5, 1.5])

        app_node("Constant",[],["S"],value=ST)
        app_node("Constant",[],["ZP"],value=ZPT)
        app_node("Constant",[],["C0"],value=C0T)

        app_node("QuantizeLinear",["X0","S","ZP"], ["QX0"])
        app_node("DequantizeLinear", ["QX0","S","ZP"], ["DQX0"])
        app_node("Resize",["DQX0","","C0"],["X1"], mode="linear")
        app_node("QuantizeLinear",["X1","S","ZP"], ["QX1"])
        app_node("DequantizeLinear", ["QX1","S","ZP"], ["DQX1"])
        app_node("Transpose", ["DQX1"], ["X2"], perm = [0, 2, 3, 1]) 
        return node_list
    
    def construct_io_value_info(self):
        input_vi = [make_tensor_value_info("X0", TensorProto.FLOAT, [1, 1, 2, 2])]
        # output_vi = [make_tensor_value_info("DQX1", TensorProto.FLOAT, [1, 1, 4, 4])]
        output_vi = [make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3, 3, 1])]
        return input_vi, output_vi

register_test(AnotherOne, "resize_tq_0")

class AnotherOne1(AnotherOne):
    def construct_nodes(self):
        node_list = super().construct_nodes()
        node_list.pop()
        node_list.append(make_node("Identity", ["DQX1"],["X2"]))
        return node_list

    def construct_io_value_info(self):
        input_vi = [make_tensor_value_info("X0", TensorProto.FLOAT, [1, 1, 2, 2])]
        output_vi = [make_tensor_value_info("X2", TensorProto.FLOAT, [1, 1, 3, 3])]
        return input_vi, output_vi

register_test(AnotherOne1, "resize_tq_1")

class AnotherOne2(AnotherOne):
    def construct_nodes(self):
        node_list = super().construct_nodes()
        node_list.pop()
        return node_list

    def construct_io_value_info(self):
        input_vi = [make_tensor_value_info("X0", TensorProto.FLOAT, [1, 1, 2, 2])]
        output_vi = [make_tensor_value_info("DQX1", TensorProto.FLOAT, [1, 1, 3, 3])]
        return input_vi, output_vi

register_test(AnotherOne2, "resize_tq_2")

class AnotherOne3(AnotherOne):
    def construct_io_value_info(self):
        input_vi = [make_tensor_value_info("X0", TensorProto.FLOAT, [1, 1, 2, 2])]
        output_vi = [make_tensor_value_info("DQX1", TensorProto.FLOAT, [1, 1, 3, 3])]
        return input_vi, output_vi

register_test(AnotherOne3, "resize_tq_3")