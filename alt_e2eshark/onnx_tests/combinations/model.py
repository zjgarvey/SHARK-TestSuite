# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy, torch, sys
import onnxruntime
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor
import os

from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test

class AltMultipleConvModel(OnnxModelInfo):
    def construct_model(self):
        # float input tensor:
        AX0 = make_tensor_value_info("AX0", TensorProto.FLOAT, [1,3,513,513])
        # quantized weight tensor inputs
        K0 = make_tensor_value_info("K0", TensorProto.FLOAT, [32, 3, 3, 3])
        K1 = make_tensor_value_info("K1", TensorProto.FLOAT, [32, 1, 3, 3])
        K2 = make_tensor_value_info("K2", TensorProto.FLOAT, [16,32,1,1])
        # output tensor
        X3 = make_tensor_value_info("X3", TensorProto.FLOAT, [1,16,257,257])

        # scale for K1
        KS1T = make_tensor("KS1T", TensorProto.FLOAT, [], [5.000000e-01])
        # ZP is a trivial zeropoint.
        ZPT = make_tensor("ZPT", TensorProto.INT8, [], [0])
        # quantized bias values and corresponding scales
        use_bias = False
        if use_bias:
            B0T = make_tensor("B0T", TensorProto.INT8, [32], [8, -58, -45, 82, 10, -40, 52, 71, 45, 81, 2, 14, 96, 49, -42, -34, 116, -32, 11, 85, 95, -51, -82, 38, 7, 60, 25, 9, 80, 59, 92, 94])
            B1T = make_tensor("B1T", TensorProto.INT8, [32], [28, -14, -5, 54, 0, 8, 44, 36, 5, 65, 39, -1, 51, 36, -14, 4, 39, -17, 32, 30, 43, -8, 2, 9, 22, 20, 43, 84, 28, 58, 72, 51])
            B2T = make_tensor("B2T", TensorProto.INT8, [16], [48, 123, -30, -68, 14, 102, -20, -18, -58, 65, -19, -12, -82, -58, -38, -59])
            BS2T = make_tensor("BS2T", TensorProto.FLOAT, [], [2.500000e-01])
        BS0T = make_tensor("BS0T", TensorProto.FLOAT, [], [3.125000e-02])
        BS1T = make_tensor("BS1T", TensorProto.FLOAT, [], [6.250000e-02]) # also used to quantize/dequantize input between convs
        use_clips = False
        if use_clips:
            LT = make_tensor("LT", TensorProto.FLOAT, [], [0.0])
            UT = make_tensor("UT", TensorProto.FLOAT, [], [6.0])

        node_list = []
        # Quantization Scheme Constants
        node_list.append(make_node("Constant",[],["ZP"],value=ZPT))
        node_list.append(make_node("Constant",[],["BS0"],value=BS0T))
        node_list.append(make_node("Constant",[],["KS1"],value=KS1T))
        node_list.append(make_node("Constant",[],["BS1"],value=BS1T))
        if use_bias:
            node_list.append(make_node("Constant",[],["BS2"],value=BS2T))
            # bias constants and conversions
            node_list.append(make_node("Constant",[],["AB0"],value=B0T))
            node_list.append(make_node("Constant",[],["AB1"],value=B1T))
            node_list.append(make_node("Constant",[],["AB2"],value=B2T))
            node_list.append(make_node("DequantizeLinear",["AB0","BS0","ZP"],["B0"]))
            node_list.append(make_node("DequantizeLinear",["AB1","BS1","ZP"],["B1"]))
            node_list.append(make_node("DequantizeLinear",["AB2","BS2","ZP"],["B2"]))
        if use_clips:
            node_list.append(make_node("Constant",[],["L"],value=LT))
            node_list.append(make_node("Constant",[],["U"],value=UT))
        node_list.append(make_node("QuantizeLinear",["AX0", "BS1", "ZP"],["BX0"]))
        node_list.append(make_node("DequantizeLinear",["BX0","BS1","ZP"],["X0"]))
        node_list.append(make_node(
            op_type="Conv",
            inputs=["X0","K0","B0"] if use_bias else ["X0","K0"],
            outputs=["AX1"],
            group=1,
            kernel_shape=[3,3],
            pads=[1,1,1,1],
            strides=[2,2],
        ))
        name1 = "AX1"
        if use_clips:
            name1 = "CX1"
            node_list.append(make_node("Clip", ["AX1","L","U"], [name1]))
        node_list.append(make_node("QuantizeLinear",[name1, "BS1", "ZP"],["BX1"]))
        node_list.append(make_node("DequantizeLinear",["BX1","BS1","ZP"],["X1"]))
        node_list.append(make_node(
            op_type="Conv",
            inputs=["X1","K1","B1"] if use_bias else ["X1","K1"],
            outputs=["AX2"],
            group=32,
            kernel_shape=[3,3],
            pads=[1,1,1,1],
            strides=[1,1],
        ))
        name2 = "AX2"
        if use_clips:
            name2 = "CX2"
            node_list.append(make_node("Clip", ["AX2","L","U"], [name2]))
        node_list.append(make_node("QuantizeLinear", [name2,"BS1","ZP"], ["BX2"]))
        node_list.append(make_node("DequantizeLinear", ["BX2","BS1","ZP"],["X2"]))
        node_list.append(make_node(
            op_type="Conv",
            inputs=["X2","K2","B2"] if use_bias else ["X2","K2"],
            outputs=["X3"],
            group=1,
            kernel_shape=[1,1],
            pads=[0,0,0,0],
            strides=[1,1],
        ))

        graph=make_graph(
            node_list,
            "main",
            [AX0, K0, K1, K2],
            [X3],
        )

        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)

register_test(AltMultipleConvModel, "multi_conv_alt")
        

class MultipleConvModel(OnnxModelInfo):
    def construct_model(self):
        # float input tensor:
        AX0 = make_tensor_value_info("AX0", TensorProto.FLOAT, [1,3,513,513])
        # quantized weight tensor inputs
        BK0 = make_tensor_value_info("BK0", TensorProto.INT8, [32, 3, 3, 3])
        BK1 = make_tensor_value_info("BK1", TensorProto.INT8, [32, 1, 3, 3])
        BK2 = make_tensor_value_info("BK2", TensorProto.INT8, [16,32,1,1])
        # output tensor
        X3 = make_tensor_value_info("X3", TensorProto.FLOAT, [1,16,257,257])

        # scale for K1
        KS1T = make_tensor("KS1T", TensorProto.FLOAT, [], [5.000000e-01])
        # ZP is a trivial zeropoint.
        ZPT = make_tensor("ZPT", TensorProto.INT8, [], [0])
        # quantized bias values and corresponding scales
        use_bias = False
        if use_bias:
            B0T = make_tensor("B0T", TensorProto.INT8, [32], [8, -58, -45, 82, 10, -40, 52, 71, 45, 81, 2, 14, 96, 49, -42, -34, 116, -32, 11, 85, 95, -51, -82, 38, 7, 60, 25, 9, 80, 59, 92, 94])
            B1T = make_tensor("B1T", TensorProto.INT8, [32], [28, -14, -5, 54, 0, 8, 44, 36, 5, 65, 39, -1, 51, 36, -14, 4, 39, -17, 32, 30, 43, -8, 2, 9, 22, 20, 43, 84, 28, 58, 72, 51])
            B2T = make_tensor("B2T", TensorProto.INT8, [16], [48, 123, -30, -68, 14, 102, -20, -18, -58, 65, -19, -12, -82, -58, -38, -59])
            BS2T = make_tensor("BS2T", TensorProto.FLOAT, [], [2.500000e-01])
        BS0T = make_tensor("BS0T", TensorProto.FLOAT, [], [3.125000e-02])
        BS1T = make_tensor("BS1T", TensorProto.FLOAT, [], [6.250000e-02]) # also used to quantize/dequantize input between convs
        use_clips = False
        if use_clips:
            LT = make_tensor("LT", TensorProto.FLOAT, [], [0.0])
            UT = make_tensor("UT", TensorProto.FLOAT, [], [6.0])

        node_list = []
        # Quantization Scheme Constants
        node_list.append(make_node("Constant",[],["ZP"],value=ZPT))
        node_list.append(make_node("Constant",[],["BS0"],value=BS0T))
        node_list.append(make_node("Constant",[],["KS1"],value=KS1T))
        node_list.append(make_node("Constant",[],["BS1"],value=BS1T))
        if use_bias:
            node_list.append(make_node("Constant",[],["BS2"],value=BS2T))
            # bias constants and conversions
            node_list.append(make_node("Constant",[],["AB0"],value=B0T))
            node_list.append(make_node("Constant",[],["AB1"],value=B1T))
            node_list.append(make_node("Constant",[],["AB2"],value=B2T))
            node_list.append(make_node("DequantizeLinear",["AB0","BS0","ZP"],["B0"]))
            node_list.append(make_node("DequantizeLinear",["AB1","BS1","ZP"],["B1"]))
            node_list.append(make_node("DequantizeLinear",["AB2","BS2","ZP"],["B2"]))
        if use_clips:
            node_list.append(make_node("Constant",[],["L"],value=LT))
            node_list.append(make_node("Constant",[],["U"],value=UT))
        node_list.append(make_node("QuantizeLinear",["AX0", "BS1", "ZP"],["BX0"]))
        node_list.append(make_node("DequantizeLinear",["BX0","BS1","ZP"],["X0"]))
        node_list.append(make_node("DequantizeLinear",["BK0","BS0","ZP"],["K0"]))
        node_list.append(make_node(
            op_type="Conv",
            inputs=["X0","K0","B0"] if use_bias else ["X0","K0"],
            outputs=["AX1"],
            group=1,
            kernel_shape=[3,3],
            pads=[1,1,1,1],
            strides=[2,2],
        ))
        name1 = "AX1"
        if use_clips:
            name1 = "CX1"
            node_list.append(make_node("Clip", ["AX1","L","U"], [name1]))
        node_list.append(make_node("QuantizeLinear",[name1, "BS1", "ZP"],["BX1"]))
        node_list.append(make_node("DequantizeLinear",["BX1","BS1","ZP"],["X1"]))
        node_list.append(make_node("DequantizeLinear",["BK1","KS1","ZP"],["K1"]))
        node_list.append(make_node(
            op_type="Conv",
            inputs=["X1","K1","B1"] if use_bias else ["X1","K1"],
            outputs=["AX2"],
            group=32,
            kernel_shape=[3,3],
            pads=[1,1,1,1],
            strides=[1,1],
        ))
        name2 = "AX2"
        if use_clips:
            name2 = "CX2"
            node_list.append(make_node("Clip", ["AX2","L","U"], [name2]))
        node_list.append(make_node("QuantizeLinear", [name2,"BS1","ZP"], ["BX2"]))
        node_list.append(make_node("DequantizeLinear", ["BX2","BS1","ZP"],["X2"]))
        node_list.append(make_node("DequantizeLinear",["BK2","BS1","ZP"],["K2"]))
        node_list.append(make_node(
            op_type="Conv",
            inputs=["X2","K2","B2"] if use_bias else ["X2","K2"],
            outputs=["X3"],
            group=1,
            kernel_shape=[1,1],
            pads=[0,0,0,0],
            strides=[1,1],
        ))

        graph=make_graph(
            node_list,
            "main",
            [AX0, BK0, BK1, BK2],
            [X3],
        )

        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)

register_test(MultipleConvModel, "multi_conv")
        

class ShapeIntoCOSCombinationModel(OnnxModelInfo):
    def construct_model(self):
        input_X = make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5])
        input_B = make_tensor_value_info("B", TensorProto.FLOAT, [3])
        output_Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None, None])

        shape_node = make_node("Shape", ["B"], ["S"], "shape_node")

        cos_node = make_node(
            "ConstantOfShape",
            ["S"],
            ["C"],
            value=onnx.numpy_helper.from_array(numpy.array([5], dtype=numpy.int64)),
        )
        expand_node = make_node("Expand", ["X", "C"], ["Y"], "expand_node")

        graph = make_graph(
            [shape_node, cos_node, expand_node],
            "main",
            [input_X, input_B],
            [output_Y],
        )

        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 11

        onnx.save(onnx_model, self.model)


register_test(ShapeIntoCOSCombinationModel, "shape_to_cos_test")
