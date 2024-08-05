# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import onnx
from onnx.helper import make_node, make_graph, make_model
from pathlib import Path
from e2e_testing.framework import OnnxModelInfo
from e2e_testing.onnx_utils import (
    modify_model_output,
    node_output_name,
    node_name_from_back,
)

"""This file contains several helpful child classes of OnnxModelInfo."""


class SiblingModel(OnnxModelInfo):
    """convenience class for re-using an onnx model from another 'sibling' test"""

    def __init__(self, og_model_info_class: type, og_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # additionally store an instance of the sibling test
        run_dir = Path(self.model).parents[1]
        og_model_path = os.path.join(run_dir, og_name)
        self.sibling_inst = og_model_info_class(og_name, og_model_path)

    def construct_model(self):
        if not os.path.exists(self.sibling_inst.model):
            self.sibling_inst.construct_model()
        self.model = self.sibling_inst.model


def get_sibling_constructor(sibling_class, og_constructor, og_name):
    """Returns a constructor for the sibling class. Useful for convenient registration.

    Usage:

    class OGModelInfoClass:
        ...

    register_test(OGModelInfoClass, og_name)

    class NewSiblingModel(SiblingModel):
        ...

    sibling_constructor = get_sibling_constructor(NewSiblingModel, OGModelInfoClass, og_name)
    register_test(sibling_constructor, new_name)

    """
    return lambda *args, **kwargs: sibling_class(
        og_constructor, og_name, *args, **kwargs
    )


class TruncatedModel(SiblingModel):
    """This class will take the model.onnx from another test, and modify the output.

    Takes additional __init__ args: n (int) and op_type (str)
    If op_type = "", n will determine the position backwards from the original output node.
    If op_type isn't a null string, then n will be used to determine which node of that op_type will be returned as an output.

    Examples:
    op_type = "Conv" and n=2: This will set the output of the onnx model to the ouptput of the third Conv node in the graph.
    op_type = "" and n=2: This will set the output of the model to the second-to-last node before the original output.
    """

    def __init__(self, n: int, op_type: str, *args, **kwargs):
        self.n = n
        self.op_type = op_type
        super().__init__(*args, **kwargs)

    def construct_model(self):
        if not os.path.exists(self.sibling_inst.model):
            self.sibling_inst.construct_model()
        og_model = onnx.load(self.sibling_inst.model)
        inf_model = onnx.shape_inference.infer_shapes(og_model, data_prop=True)
        output_name = (
            node_name_from_back(inf_model, self.n)
            if self.op_type == ""
            else node_output_name(inf_model, self.n, self.op_type)
        )
        new_model = modify_model_output(inf_model, output_name)
        onnx.save(new_model, self.model)


def get_trucated_constructor(truncated_class, og_constructor, og_name):
    """returns a function that takes in (n, op_type) and returns a constructor for the truncated class.

    Usage:

    class OGModelInfoClass:
        ...

    register_test(OGModelInfoClass, og_name)

    class NewTruncatedModel(TruncatedModel):
        ...

    truncated_constructor = get_truncated_constructor(NewTruncatedModel, OGModelInfoClass, og_name)
    register_test(truncated_constructor(2, "Conv"), og_name + "_2_Conv")
    register_test(truncated_constructor(5, ""), og_name + "_5")

    """
    return lambda n, op_type: (
        lambda *args, **kwargs: truncated_class(
            n, op_type, og_constructor, og_name, *args, **kwargs
        )
    )


class BuildAModel(OnnxModelInfo):
    """Convenience class for building an onnx model from scratch.
    If inheriting from this class:
    1. override construct_nodes(self) to add to the self.node_list. The get_app_node method may be helpful.
    2. override construct_i_o_value_info to add the input and output value infos to the lists self.input_vi and self.output_vi.
    3. optionally override other OnnxModelInfo methods if desired, e.g. construct_inputs.

    Example:

    class QuantizedRelu(BuildAModel):
        def construct_nodes(self):
            app_node = self.get_app_node()

            ST = make_tensor("ST", TensorProto.FLOAT, [], [0.025])
            ZPT = make_tensor("ZPT", TensorProto.INT8, [], [3])

            app_node("Constant", [], ["S"], value=ST)
            app_node("Constant", [], ["ZP"], value=ZPT)
            app_node("QuantizeLinear", ["X", "S", "ZP"], ["QX"])
            app_node("DequantizeLinear", ["QX", "S", "ZP"], ["DQX"])
            app_node("Relu", ["DQX"], ["Y"])
            app_node("QuantizeLinear", ["Y", "S", "ZP"], ["QY"])
            app_node("DequantizeLinear", ["QY", "S", "ZP"], ["DQY"])

        def construct_i_o_value_info(self):
            self.input_vi.append(make_tensor_value_info("X", TensorProto.FLOAT, [1,2,4]))
            self.output_vi.append(make_tensor_value_info("DQY", TensorProto.FLOAT, [1,2,4]))

    register_test(QuantizedRelu, "quantized_relu")

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # always construct the onnx model for these tests.
        # Useful for trial-error building of the test without having to delete the old onnx model.
        self.node_list = []
        self.input_vi = []
        self.output_vi = []
        self.construct_model()

    def construct_nodes(self):
        """Needs to be overriden. Update self.node_list here with the nodes you want in your graph."""
        raise NotImplementedError("Please implement a construct_nodes method.")

    def construct_i_o_value_info(self):
        """Needs to be overridden. Update self.input_vi and self.output_vi with the lists of input and output value infos"""
        raise NotImplementedError("Please implement a construct_i_o_value_info method.")

    def get_app_node(self):
        """Convenience function for defining a lambda that appends a new node to self.node_list"""
        return lambda op_type, inputs, outputs, **kwargs: self.node_list.append(
            make_node(op_type, inputs, outputs, **kwargs)
        )

    def construct_model(self):
        self.construct_nodes()
        self.construct_i_o_value_info()
        graph = make_graph(self.node_list, "main", self.input_vi, self.output_vi)
        model = make_model(graph)
        onnx.save(model, self.model)