# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from e2e_testing.test_configs.mlir_configbase import MlirConfigBase
from e2e_testing.framework import PytorchModelInfo, Module
from typing import Tuple

class PytorchTestConfig(MlirConfigBase):
    '''This is the basic testing configuration for onnx models. This should be initialized with a specific backend, and uses torch-mlir to import the onnx model to torch-onnx MLIR, and apply torch-mlir pre-proccessing passes if desired.'''
    def import_model(self, model_info: PytorchModelInfo, *, save_to: str = None) -> Tuple[Module, str]:
        raise NotImplementedError("TODO: Implement variations of pytorch -> mlir import.")