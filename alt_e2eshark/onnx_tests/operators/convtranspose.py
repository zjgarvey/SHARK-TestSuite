# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_tensor_value_info, make_tensor

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_with_name, register_test


class ConvTransposeAutoPadSymmetric(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3]),
            make_tensor_value_info("W", TensorProto.FLOAT, [1, 2, 3, 3]),
        ]
        self.output_vi = [
            make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 8, 8]),
        ]

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node(
            "ConvTranspose",
            ["X", "W"],
            ["Y"],
            output_shape=[8, 8],
            auto_pad="SAME_UPPER",
            strides=[2, 2],
        )


register_test(ConvTransposeAutoPadSymmetric, "conv_transpose_symmetric_autopad")


def get_class_for_shapes(input_shape, kernel_shape, output_shape):
    class ConvTransposeAutoPadSymmetric(BuildAModel):
        def construct_i_o_value_info(self):
            self.input_vi = [
                make_tensor_value_info("X", TensorProto.FLOAT, input_shape),
                make_tensor_value_info("W", TensorProto.FLOAT, kernel_shape),
            ]
            self.output_vi = [
                make_tensor_value_info("Y", TensorProto.FLOAT, output_shape),
            ]

        def construct_nodes(self):
            app_node = self.get_app_node()
            app_node(
                "ConvTranspose",
                ["X", "W"],
                ["Y"],
                output_shape=output_shape[-2:],
                auto_pad="SAME_UPPER",
                strides=[2, 2],
            )

    return ConvTransposeAutoPadSymmetric


register_test(
    get_class_for_shapes([1, 1, 3, 3], [1, 2, 3, 3], [1, 2, 8, 8]), "another_sample"
)
