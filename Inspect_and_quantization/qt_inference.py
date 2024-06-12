import torch
from model.yolov4 import *
import cv2
import numpy as np
from numpy import random

import torchvision
from torchvision import tv_tensors
from torchvision.io import read_image

import time
import onnx
import onnxruntime as ort


from DPU_inference.dpu_utils import process_preds, non_max_suppression
import sys


NO_OF_ANCHOR_BOX = N = 3

NO_OF_CLASS = C = 20
HEIGHT = H = 416
WIDTH = W = 416


DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = batch_size = 16


SCALE = [32, 16, 8]
S = [13, 26, 52]

ANCHORS = (
    np.array(
        [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        ]
    )
    * np.array([[S]]).T
)


classes = [
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "diningtable",
    "pottedplant",
    "sofa",
    "tvmonitor",
]


MEANS = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def plot(final_out, image_path):
    bboxes, pred_conf, pred_labels = final_out

    im = cv2.imread(image_path)
    unique_labels = np.unique(pred_labels)

    n_cls_preds = len(unique_labels)
    bbox_colors = {
        int(cls_pred): (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for cls_pred in unique_labels
    }

    for bbox, conf, cls_pred in zip(bboxes, pred_conf, pred_labels):
        x1, y1, x2, y2 = bbox

        color = bbox_colors[int(cls_pred)]

        # Rescale coordinates to original dimensions
        ori_h, ori_w, _ = im.shape
        pre_h, pre_w = H, W
        box_h = ((y2 - y1) / pre_h) * ori_h
        box_w = ((x2 - x1) / pre_w) * ori_w
        y1 = (y1 / pre_h) * ori_h
        x1 = (x1 / pre_w) * ori_w

        # Create a Rectangle patch
        cv2.rectangle(
            im, (int(x1), int(y1)), (int(x1 + box_w), int(y1 + box_h)), color, 2
        )

        # Add label
        label = classes[int(cls_pred)] + "  " + str(conf)
        cv2.putText(
            im,
            label,
            (int(x1) + 5, int(y1) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # # Save generated image with detections
    # output_path = "prediction.jpg"
    # cv2.imwrite(output_path, im)

    # Display image
    print("Image plotted")
    cv2.imshow("Prediction", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_image(img, conf=0.5):

    onnx_model = onnx.load("qt_yolov4.onnx")
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession("qt_yolov4.onnx")
    input_name = ort_sess.get_inputs()[0].name
    
    outputs = ort_sess.run(None, {input_name: img.numpy()})

    output = [o.detach().numpy() for o in output]
    output_list = process_preds(output, S=S, SCALE=SCALE, anchor_boxes=ANCHORS)

    filtered_outputs = []
    for output in output_list:
        filtered_outputs.append(output[output[..., 0] >= conf])

    output_arr = np.concatenate(filtered_outputs, axis=0)

    final_out = non_max_suppression(output_arr, iou_threshold=0.3)

    return final_out


def main(argv):

    image_path = argv[1]
    img = read_image(image_path)
    img = torchvision.transforms.functional.resize(img, (416, 416))
    img = img / 255 - MEANS.unsqueeze(1).unsqueeze(1)
    img /= STD.unsqueeze(1).unsqueeze(1)
    img = img.unsqueeze(dim=0)

    start_time = time.time()
    # Your code to measure
    predictions = predict_image(img)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    plot(predictions, image_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage : python3 inference.py <image_path>")
    else:
        main(sys.argv)
