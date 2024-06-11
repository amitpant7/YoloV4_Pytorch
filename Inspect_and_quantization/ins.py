import torch
import sys
import os
from pytorch_nndct.apis import Inspector
from model.yolov4 import *

# To import from scirpts need this
# To import from scripts, use relative import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


target = "DPUCZDX8G_ISA1_B4096"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load("model.pth", map_location=device)

# Random Input
random_input = torch.randn(1, 3, 416, 416)

# inspection
inspector = Inspector(target)
inspector.inspect(model, random_input, device, output_dir="inspect", image_format="png")
