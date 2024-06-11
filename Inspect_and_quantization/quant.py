import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision

from model.yolov4 import *
from .vocdata import *  # or vocdata

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()


# commands to run
# python quant.py --quant_mode calib --subset_len 200
# python quant.py --quant_mode test --subset_len 1 --batch_size=1 --deploy

parser.add_argument(
    "--subset_len",
    default=200,
    type=int,
    help="subset_len to evaluate model, using the whole validation dataset if it is not set",
)

parser.add_argument(
    "--batch_size", default=4, type=int, help="input data batch size to evaluate model"
)

parser.add_argument(
    "--quant_mode",
    default="calib",
    choices=["float", "calib", "test"],
    help="quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model",
)


parser.add_argument(
    "--deploy", dest="deploy", action="store_true", help="export xmodel for deployment"
)


args, _ = parser.parse_known_args()


dataset = VOCDataset(root="./data", year="2007", image_set="train", transform=transform)

val_dataset = torch.utils.data.Subset(
    dataset, random.sample(range(0, len(dataset)), args.subset_len)
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
)


def quantization(
    title="optimize", model_name="yolov3", file_path="", val_loader=val_loader
):

    quant_mode = args.quant_mode
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len

    if quant_mode != "test" and deploy:
        deploy = False
        print(
            r"Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!"
        )
    if deploy and (batch_size != 1 or subset_len != 1):
        print(
            r"Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!"
        )
        batch_size = 1
        subset_len = 1

    model = torch.load(file_path, map_location=device)

    input = torch.randn([batch_size, 3, 416, 416])

    if quant_mode == "float":
        quant_model = model

    else:

        quantizer = torch_quantizer(quant_mode, model, (input), device=device)
        quant_model = quantizer.quant_model

    for images, bboxes, labels in tqdm(val_loader, leave=False):
        images = images.to(device)
        with torch.no_grad():
            quant_model(images)

    # handle quantization result
    if quant_mode == "calib":
        # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel()


if __name__ == "__main__":

    model_name = "yolov4"
    file_path = "yolov4_replace_finetune_wts.pth"

    feature_test = " float model evaluation"
    if args.quant_mode != "float":
        feature_test = " quantization"
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += " with optimization"
    else:
        feature_test = " float model evaluation"
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    # calibration or evaluation
    quantization(title=title, model_name=model_name, file_path=file_path)

    print("-------- End of {} test ".format(model_name))
