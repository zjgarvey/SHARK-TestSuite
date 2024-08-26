# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from e2e_testing.backends import BackendBase
from e2e_testing.framework import TestConfig, Module, CompiledArtifact
from e2e_testing.storage import TestTensors
from typing import Tuple

REDUCE_TO_LINALG_PIPELINE = [
    "torch-lower-to-backend-contract",
    "torch-backend-to-linalg-on-tensors-backend-pipeline",
]

class MlirConfigBase(TestConfig):
    """A base configuration for passing load/compile to a backend and running torch-mlir preprocessing."""
    def __init__(
        self, log_dir: str, backend: BackendBase, torch_mlir_pipeline: Tuple[str, ...]
    ):
        super().__init__()
        self.log_dir = log_dir
        self.backend = backend
        if len(torch_mlir_pipeline) > 0:
            self.pass_pipeline = "builtin.module(" + ",".join(torch_mlir_pipeline) + ")"
        else:
            self.pass_pipeline = None

    def preprocess_model(self, mlir_module: Module, *, save_to: str = None) -> Module:
        # if the pass pipeline is empty, return the original module
        if not self.pass_pipeline:
            return mlir_module
        # convert imported torch-onnx ir to torch
        onnx_to_torch_pipeline = "builtin.module(func.func(convert-torch-onnx-to-torch))"
        from torch_mlir.passmanager import PassManager
        with mlir_module.context as ctx:
            pm0 = PassManager.parse(onnx_to_torch_pipeline)
            pm0.run(mlir_module.operation)
            # log torch-mlir IR
            if save_to:
                with open(save_to + "model.torch.mlir", "w") as f:
                    f.write(str(mlir_module))
            pm1 = PassManager.parse(self.pass_pipeline)
            pm1.run(mlir_module.operation)
            # log modified IR
            if save_to:
                with open(save_to + "model.modified.mlir", "w") as f:
                    f.write(str(mlir_module))
        return mlir_module

    def compile(self, mlir_module: Module, *, save_to: str = None) -> CompiledArtifact:
        return self.backend.compile(mlir_module, save_to=save_to)

    def run(self, artifact: CompiledArtifact, inputs: TestTensors, *, func_name="main") -> TestTensors:
        func = self.backend.load(artifact, func_name=func_name)
        return func(inputs)

