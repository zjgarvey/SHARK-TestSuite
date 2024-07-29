# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import onnx
import numpy
import torch
import urllib
from pathlib import Path
from PIL import Image
from torchvision import transforms
from e2e_testing.framework import SiblingModel
from e2e_testing.onnx_utils import modify_model_output, node_output_name, node_name_from_back
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors
from .azure_models import AzureDownloadableModel

label_map = numpy.array([
    (0, 0, 0),  # background
    (128, 0, 0),  # aeroplane
    (0, 128, 0),  # bicycle
    (128, 128, 0),  # bird
    (0, 0, 128),  # boat
    (128, 0, 128),  # bottle
    (0, 128, 128),  # bus
    (128, 128, 128),  # car
    (64, 0, 0),  # cat
    (192, 0, 0),  # chair
    (64, 128, 0),  # cow
    (192, 128, 0),  # dining table
    (64, 0, 128),  # dog
    (192, 0, 128),  # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),  # potted plant
    (128, 64, 0),  # sheep
    (0, 192, 0),  # sofa
    (128, 192, 0),  # train
    (0, 64, 128),  # tv/monitor
])

class DeeplabModel(SiblingModel):
    def construct_inputs(self):
        filename = str(Path(self.model).parent.joinpath("input.png"))
        url = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png"
        try: urllib.URLopener().retrieve(url, filename)
        except: urllib.request.urlretrieve(url, filename)
        input_image = Image.open(filename)
        input_image = input_image.convert("RGB")
        self.img_size = input_image.size
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((513,513)),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).transpose(1,2).transpose(2,3)
        return TestTensors((input_batch,))

    def apply_postprocessing(self, output: TestTensors):
        processed_outputs = []
        for d in output.to_torch().data:
            c = torch.topk(torch.nn.functional.softmax(d, -1), 2)[-1][0,:,:,-1]
            image = numpy.zeros([513,513,3]).astype(numpy.uint8)
            for i in range(513):
                for j in range(513):
                    for k in range(3):
                        image[i,j,k] = label_map[c[i,j]][k]
            processed_outputs.append(image)
        return TestTensors(processed_outputs)
    
    def save_processed_output(self, output: TestTensors, save_to: str, name: str):
        c = 0
        for d in output.to_numpy().data:
            im = Image.fromarray(d)
            if self.img_size:
                im = im.resize(self.img_size)
            fp = save_to + name + "." + str(c) + ".jpeg"
            im.save(fp)
            c += 1

# base test (no post-processing or input mods)
register_test(AzureDownloadableModel, "deeplabv3")

# sibling test with all the bells & whistles
constructor = lambda *args, **kwargs : DeeplabModel(AzureDownloadableModel, "deeplabv3", *args, **kwargs)
register_test(constructor, "deeplabv3_real_with_pp")


class TruncatedDeeplabModel(SiblingModel):
    def __init__(self, n: int, op_type: str, *args, **kwargs):
        self.n = n
        self.op_type = op_type
        super().__init__(*args, **kwargs)

    def construct_model(self):
        if not os.path.exists(self.sibling_inst.model):
            self.sibling_inst.construct_model()
        og_model = onnx.load(self.sibling_inst.model)
        inf_model = onnx.shape_inference.infer_shapes(og_model, data_prop=True)
        output_name = node_name_from_back(inf_model, self.n) if self.op_type == "" else node_output_name(inf_model, self.n, self.op_type)
        new_model = modify_model_output(inf_model, output_name)
        onnx.save(new_model, self.model)

    def construct_inputs(self):
        filename = str(Path(self.model).parent.joinpath("input.png"))
        url = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png"
        try: urllib.URLopener().retrieve(url, filename)
        except: urllib.request.urlretrieve(url, filename)
        input_image = Image.open(filename)
        input_image = input_image.convert("RGB")
        self.img_size = input_image.size
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((513,513)),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).transpose(1,2).transpose(2,3)
        return TestTensors((input_batch,))

trunc_constructor = lambda n, op_type : (lambda *args, **kwargs : TruncatedDeeplabModel(n, op_type, AzureDownloadableModel, "deeplabv3", *args, **kwargs))

# register_test(trunc_constructor(2, "Conv"), "deeplabv3_2_Conv")
# register_test(trunc_constructor(0, "Add"), "deeplabv3_0_Add")
# register_test(trunc_constructor(0, "Mul"), "deeplabv3_0_Mul")
# register_test(trunc_constructor(0, "Relu"), "deeplabv3_0_Relu")
# register_test(trunc_constructor(1, "Relu"), "deeplabv3_1_Relu")
# register_test(trunc_constructor(0, "Resize"), "deeplabv3_0_Resize")
# register_test(trunc_constructor(1, "Resize"), "deeplabv3_1_Resize")
for n in range(1, 7):
    register_test(trunc_constructor(n, ""), f"deeplabv3_{n}_fb")

