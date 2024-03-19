# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing add
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [])
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [6])

# Create an output
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [6])

# Create a node (NodeProto)
addnode = make_node(
    "Add", ["X", "Y"], ["Z"], "addnode"  # node name  # inputs  # outputs
)

# Create the graph (GraphProto)
graph = make_graph(
    [addnode],
    "addgraph",
    [X, Y],
    [Z],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 19

# Save the model
# save_model(onnx_model, "model.onnx")
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


session = onnxruntime.InferenceSession("model.onnx", None)
model_input_Y = numpy.random.randn(6).astype(numpy.float32)
model_input_X = numpy.random.randn()
model_input_X = numpy.array(numpy.float32(model_input_Y))
# gets X in inputs[0] and Y in inputs[1]
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_X, inputs[1].name: model_input_Y},
)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
E2ESHARK_CHECK["input"] = [
    torch.from_numpy(model_input_X),
    torch.from_numpy(model_input_Y),
]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file constitutes end part of runmodel.py
# this is appended to the model.py in test dir

import numpy
import onnxruntime
import sys, argparse
import torch, pickle
from commonutils import getOutputTensorList, E2ESHARK_CHECK_DEF, postProcess

msg = "The script to run an ONNX model test"
parser = argparse.ArgumentParser(description=msg, epilog="")

parser.add_argument(
    "-d",
    "--todtype",
    choices=["default", "fp32", "fp16", "bf16"],
    default="default",
    help="If not default, casts model and input to given data type if framework supports model.to(dtype) and tensor.to(dtype)",
)
parser.add_argument(
    "-m",
    "--mode",
    choices=["direct", "onnx", "ort"],
    default="direct",
    help="Generate torch MLIR, ONNX or ONNX plus ONNX RT stub",
)
parser.add_argument(
    "-o",
    "--outfileprefix",
    default="model",
    help="Prefix of output files written by this model",
)
args = parser.parse_args()
if args.todtype != "default":
    print(
        "Onnx does not support model.to(dtype). Default dtype of the model will be used."
    )

runmode = args.mode
outfileprefix = args.outfileprefix
outfileprefix += "." + args.todtype
inputsavefilename = outfileprefix + ".input.pt"
outputsavefilename = outfileprefix + ".goldoutput.pt"


E2ESHARK_CHECK["postprocessed_output"] = postProcess(E2ESHARK_CHECK)
# TBD, remobe torch.save and use the .pkl instead
torch.save(E2ESHARK_CHECK["input"], inputsavefilename)
torch.save(E2ESHARK_CHECK["output"], outputsavefilename)

# Save the E2ESHARK_CHECK
with open("E2ESHARK_CHECK.pkl", "wb") as f:
    pickle.dump(E2ESHARK_CHECK, f)
