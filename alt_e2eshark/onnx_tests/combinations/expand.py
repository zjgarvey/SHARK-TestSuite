# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import (
    make_tensor_value_info,
    make_tensor,
)
from typing import Optional
import numpy as np
import onnxruntime
from pathlib import Path

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test


class UnOptimizedExpand(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 1, 2])]
        self.output_vi = [make_tensor_value_info("Y", TensorProto.FLOAT, [])]

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node("Identity", ["Four"], ["Four_Copy"])
        app_node(
            "ConstantOfShape",
            ["Four_Copy"],
            ["Splat_2"],
            value=make_tensor("Two", TensorProto.INT64, [1], [2]),
        )
        app_node("Expand", ["X", "Splat_2"], ["Y"])
        # app_node("Shape", ["Y"], ["YShape"])
        # app_node("Shape", ["YShape"], ["Z"])

    def construct_initializers(self):
        self.initializers = [
            make_tensor("Four", TensorProto.INT64, [1], [4]),
        ]


register_test(UnOptimizedExpand, "expand_unoptimized")


class WithBasicOpt(UnOptimizedExpand):
    def apply_ort_basic_optimizations(self):
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        optimized_model = str(Path(self.model).parent.joinpath("model.optimized.onnx"))
        opt.optimized_model_filepath = optimized_model
        session = onnxruntime.InferenceSession(self.model, opt)
        self.model = optimized_model
        del session

    def construct_model(self):
        super().construct_model()
        self.apply_ort_basic_optimizations()


register_test(WithBasicOpt, "expand_optimized")
