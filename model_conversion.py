from onnx_donut.exporter import export_onnx
from onnx_donut.predictor import OnnxPredictor
from onnx_donut.quantizer import quantize
import numpy as np
from PIL import Image

# Hugging Face model card or folder
model_path = "naver-clova-ix/donut-base-finetuned-docvqa"

# Folder where the exported model will be stored
# make necessary changes for custom path
dst_folder = "converted_donut"

# Folder where the exported int8 model will be stored
dst_folder_int8 = "converted_donut_int8"

# Export from Pytorch to ONNX
export_onnx(model_path, dst_folder, opset_version=16)
print("Conversion to ONNX model Done...")

# Quantize your model to int8
quantize(dst_folder, dst_folder_int8)
print("Quantize to ONNX model int8 Done...")
