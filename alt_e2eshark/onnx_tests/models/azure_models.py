# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path
from ..helper_classes import AzureDownloadableModel
from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file
import onnxruntime

this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")

model_names = load_test_txt_file(lists_dir.joinpath("shark-test-suite.txt"))
for i in range(1,4):
    model_names += load_test_txt_file(lists_dir.joinpath(f"vai-hf-cnn-fp32-shard{i}.txt"))
    model_names += load_test_txt_file(lists_dir.joinpath(f"vai-int8-p0p1-shard{i}.txt"))
model_names += load_test_txt_file(lists_dir.joinpath("vai-vision-int8.txt"))

custom_registry = [
    "opt-125M-awq",
    "opt-125m-gptq",
    "DeepLabV3_resnet50_vaiq_int8",
]

# for simple models without dim params or additional customization, we should be able to register them directly with AzureDownloadableModel
# TODO: many of the models in the text files loaded from above will likely need to be registered with an alternative test info class.
for t in set(model_names).difference(custom_registry):
    register_test(AzureDownloadableModel, t)

from ..helper_classes import TruncatedModel, get_trucated_constructor, SiblingModel, get_sibling_constructor

# Expand example

class WithOpt(SiblingModel):
    def apply_ort_basic_optimizations(self):
        opt = onnxruntime.SessionOptions()
        opt.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        optimized_model = str(Path(self.run_dir).joinpath(self.name, "model.onnx"))
        opt.optimized_model_filepath = optimized_model
        session = onnxruntime.InferenceSession(self.model, opt)
        self.model = optimized_model
        del session

    def construct_model(self):
        super().construct_model()
        self.apply_ort_basic_optimizations()

class TruncatedWithOpt(TruncatedModel):
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

from .migraphx import misc_models, dim_param_constructor

register_test(get_sibling_constructor(WithOpt, AzureDownloadableModel, "gcvit_xxtiny"), "gcvit_xxtiny_opt")
names = ["migraphx_bert__bertsquad-12", "migraphx_models__whisper-tiny-decoder", "migraphx_models__whisper-tiny-encoder"]
for n in names:
    register_test(get_sibling_constructor(WithOpt, dim_param_constructor(misc_models[n]), n), f"{n}_opt")
exp_trunc = get_trucated_constructor(TruncatedModel, AzureDownloadableModel, "gcvit_xxtiny")
exp_trunc_opt = get_trucated_constructor(TruncatedWithOpt, AzureDownloadableModel, "gcvit_xxtiny")
register_test(exp_trunc(0, "Expand"), "gcvit_xxtiny_expand_example_unopt")
register_test(exp_trunc_opt(0, "Expand"), "gcvit_xxtiny_expand_example_opt")

