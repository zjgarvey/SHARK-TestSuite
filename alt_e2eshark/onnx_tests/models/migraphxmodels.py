from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test
import onnx
import os
import shutil
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed, fix_output_shapes
from onnxconverter_common import float16

class ExplicitPathModel(OnnxModelInfo):
    def __init__(self, external_model: str, quantization: str = "FP32", dim_params: dict | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.external_model = external_model
        self.quantization = quantization
        self.dim_param_dict = dim_params
        self.opset_version = 21
    
    def construct_model(self):
        if self.quantization == "INT8":
            #quantize self.external_model to INT8
            quantize_dynamic(self.external_model, self.model)
        elif self.quantization == "FP16":
            # convert self.external_model to FP16
            old_model = onnx.load(self.external_model)
            model = float16.convert_float_to_float16(old_model)
            onnx.save(model, self.model)
        else:
            shutil.copy(self.external_model, self.model)

path_prefix_0 = "/mnt/nas_share/migraphx/models/"
path_prefix_ORT = os.path.join(path_prefix_0, "ORT")
path_prefix_onnx_models = os.path.join(path_prefix_ORT, "onnx_models")
path_prefix_zoo = os.path.join(path_prefix_0, "onnx-model-zoo")
path_prefix_mlperf = os.path.join(path_prefix_0, "mlperf")

prefix_name_dict = dict()
ORT_list = [
    "bert_base_cased_1",
    "bert_base_uncased_1",
    "bert_large_uncased_1",
    "distilgpt2_1",
]
onnx_models_list = [
    "onnx_models/bert_base_cased_1_fp16_gpu",
    "onnx_models/bert_large_uncased_1_fp16_gpu",
    "onnx_models/distilgpt2_1_fp16_gpu",
]
zoo_list = ["gpt2-10"]
mlperf_list = ["resnet50_v1"]

dim_params = {
    "unk__616" : 1,
    "batch_size" : 1,
    "seq_len" : 10,
    "Addoutput_1_dim_2" : 1,
}


generator = lambda prefix, name : (lambda *args, **kwargs : ExplicitPathModel(os.path.join(prefix,name)+".onnx", "FP32", dim_params, *args, **kwargs))


for name in ORT_list:
    register_test(generator(path_prefix_ORT, name), name + "_basic")

for name in onnx_models_list:
    register_test(generator(path_prefix_onnx_models, name), name + "_basic")

for name in zoo_list:
    register_test(generator(path_prefix_zoo, name), name + "_basic")

for name in mlperf_list:
    register_test(generator(path_prefix_mlperf, name), name + "_basic")
