# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import onnx
from e2e_testing.test_configs.mlir_configbase import MlirConfigBase
from e2e_testing.backends import BackendBase
from e2e_testing.framework import TestConfig, TestModel, OnnxModelInfo, PytorchModelInfo, Module
from e2e_testing.storage import TestTensors
from e2e_testing.onnx_utils import pytorch_to_onnx
from typing import Tuple
from onnxruntime import InferenceSession

class OnnxEpTestConfig(TestConfig):
    '''This is the basic testing configuration for onnx models'''
    def __init__(self, log_dir: str, backend: BackendBase):
        super().__init__()
        self.log_dir = log_dir
        self.backend = backend

    def import_model(self, model_info: OnnxModelInfo, *, save_to: str = None) -> Tuple[onnx.ModelProto, None]:
        model = onnx.load(model_info.model)
        if model_info.opset_version:
            model = onnx.version_converter.convert_version(
                model, model_info.opset_version
            )
        # don't save the model, since it already exists in the log directory.
        return model, None
    
    def preprocess_model(self, model: onnx.ModelProto, *, save_to: str) -> onnx.ModelProto:
        shaped_model = onnx.shape_inference.infer_shapes(model, data_prop=True)
        if save_to:
            onnx.save(shaped_model, save_to + "inferred_model.onnx")
        return shaped_model

    def compile(self, model: onnx.ModelProto, *, save_to: str = None) -> InferenceSession:
        return self.backend.compile(model, save_to=save_to)

    def run(self, session: InferenceSession, inputs: TestTensors, *, func_name=None) -> TestTensors:
        func = self.backend.load(session)
        return func(inputs)


class OnnxTestConfig(MlirConfigBase):
    '''This is the basic testing configuration for onnx models. This should be initialized with a specific backend, and uses torch-mlir to import the onnx model to torch-onnx MLIR, and apply torch-mlir pre-proccessing passes if desired.'''
    def import_model(self, model_info: TestModel, *, save_to: str = None) -> Tuple[Module, str]:
        if isinstance(model_info, PytorchModelInfo):
            model = pytorch_to_onnx(model_info)
            opset_version = None
        elif isinstance(model_info, OnnxModelInfo):
            model = onnx.load(model_info.model)
            opset_version = model_info.opset_version
        if opset_version:
            model = onnx.version_converter.convert_version(
                model, opset_version
            )
        shaped_model = onnx.shape_inference.infer_shapes(model, data_prop=True)
        func_name = shaped_model.graph.name

        from torch_mlir.extras import onnx_importer
        from torch_mlir.dialects import torch as torch_d
        from torch_mlir.ir import Context
        context = Context()
        torch_d.register_dialect(context)
        model_info = onnx_importer.ModelInfo(shaped_model)
        m = model_info.create_module(context=context)
        imp = onnx_importer.NodeImporter.define_function(
            model_info.main_graph, m.operation
        )
        imp.import_all()
        # log imported IR
        if save_to:
            with open(save_to + "model.torch_onnx.mlir", "w") as f:
                f.write(str(m))
        return m, func_name