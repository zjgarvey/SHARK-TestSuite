# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from e2e_testing.framework import PytorchModelInfo
from e2e_testing.storage import TestTensors
from e2e_testing.registry import register_test

# hugging face example:
class Opt125MGPTQModelInfo(PytorchModelInfo):
    def __init__(self, name: str, dir_path: str):
        super().__init__(name, dir_path)
        test_modelname = "facebook/opt-125m"
        self.tokenizer = AutoTokenizer.from_pretrained(test_modelname)
    
    def get_signature(self, *, from_inputs=True):
        return [[[-1,-1]],[torch.int64]]

    def construct_model(self):
        # model origin: https://huggingface.co/jlsilva/facebook-opt-125m-gptq4bit
        quantizedmodelname = "jlsilva/facebook-opt-125m-gptq4bit"
        kwargs = {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
        }

        quantization_config = GPTQConfig(bits=8, disable_exllama=True)
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            quantizedmodelname, **kwargs
        )
        # model.output_hidden_states = False

    def construct_inputs(self):
        self.prompt = "What is nature of our existence?"
        self.encoding = self.tokenizer(self.prompt, return_tensors="pt")
        return TestTensors(
            (self.encoding["input_ids"],)
        )
    
    def apply_postprocessing(self, output: TestTensors):
        return TestTensors(torch.nn.functional.softmax(output.data[0], -1), )
    
    def save_processed_output(self, output: TestTensors, save_to: str, name: str):
        response = self.tokenizer.decode(output.data[0]).encode("utf-8")
        contents = f"Prompt: {self.prompt}\n"
        contents += f"Response: {response}"
        from pathlib import Path
        with open(Path(save_to).joinpath(name + ".txt"), "w") as file:
            file.write(contents)

        # model_response = model.generate(
        #     E2ESHARK_CHECK["input"],
        #     do_sample=True,
        #     top_k=50,
        #     max_length=100,
        #     top_p=0.95,
        #     temperature=1.0,
        # )
        # print("Response:", tokenizer.decode(model_response[0]))


register_test(Opt125MGPTQModelInfo, "opt-125m-gptq")

class SampleBasicMatmul(PytorchModelInfo):
    def get_signature(self, *, from_inputs=True):
        return [[[2, 5],[5, 3]],[torch.float32, torch.float32]]
    
    def construct_inputs(self) -> TestTensors:
        input_list = []
        