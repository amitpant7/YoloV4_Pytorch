import sys
import xir
import vart
import time
import threading
from typing import List
from ctypes import *
import random
import os

import cv2
import importlib
import numpy as np

from dpu_utils import process_preds, non_max_suppression


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
)  # Scaling up to S range

CLASSES = [
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

W, H = 416, 416

MEANS = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def runYolo(dpu_runner, image, image_path, name, conf=0.85):

    inputTensors = dpu_runner.get_input_tensors()  #  get the model input tensor
    outputTensors = dpu_runner.get_output_tensors()  # get the model ouput tensor

    # shape of output (B, H, W, A, 5+C)
    outputHeight_0 = outputTensors[0].dims[1]
    outputWidth_0 = outputTensors[0].dims[2]
    outputanchors_0 = outputTensors[0].dims[3]
    outputpreds_0 = outputTensors[0].dims[4]

    outputHeight_1 = outputTensors[1].dims[1]
    outputWidth_1 = outputTensors[1].dims[2]
    outputanchors_1 = outputTensors[1].dims[3]
    outputpreds_1 = outputTensors[1].dims[4]

    outputHeight_2 = outputTensors[2].dims[1]
    outputWidth_2 = outputTensors[2].dims[2]
    outputanchors_2 = outputTensors[2].dims[3]
    outputpreds_2 = outputTensors[2].dims[4]

    outputSize_0 = [outputHeight_0, outputWidth_0, outputanchors_0, outputpreds_0]
    outputSize_1 = [outputHeight_1, outputWidth_1, outputanchors_1, outputpreds_1]
    outputSize_2 = [outputHeight_2, outputWidth_2, outputanchors_2, outputpreds_2]

    runSize = 1
    shapeIn = (runSize,) + tuple(
        [inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:]
    )

    """prepare batch input/output """
    outputData = []
    inputData = []

    outputData.append(
        np.empty(
            tuple([runSize] + outputSize_0),
            dtype=np.float32,
            order="C",
        )
    )
    outputData.append(
        np.empty(
            tuple([runSize] + outputSize_1),
            dtype=np.float32,
            order="C",
        )
    )
    outputData.append(
        np.empty(
            tuple([runSize] + outputSize_2),
            dtype=np.float32,
            order="C",
        )
    )

    # input should also be list.
    inputData.append(np.empty((shapeIn), dtype=np.float32, order="C"))

    """init input image to input buffer """

    imageRun = inputData[0]
    imageRun[0, ...] = image.reshape(
        inputTensors[0].dims[1],
        inputTensors[0].dims[2],
        inputTensors[0].dims[3],
    )

    """Execute Async"""
    job_id = dpu_runner.execute_async(inputData, outputData)
    dpu_runner.wait(job_id)

    """Post Processing"""

    # processing rawoutputs of model and converting from tensors(in range 0-1) to pixel values for bb

    outputData = reversed(outputData)
    output_list = process_preds(outputData, S, SCALE, anchor_boxes=ANCHORS)

    # filtering outputs based on confidance
    filtered_outputs = []
    for output in output_list:
        filtered_outputs.append(output[output[..., 0] >= conf])

    output_arr = np.concatenate(filtered_outputs, axis=0)
    # print("output after concat:", output_arr.shape)

    # Perform Non Max Supression

    bboxes, pred_conf, pred_labels = non_max_suppression(output_arr, iou_threshold=0.4)

    print("Output Boxes:", bboxes, "Confidance:", pred_conf, "Classes:", pred_labels)

    # """Plot prediction with bounding box"""
    classes = CLASSES

    print("Boxes:\n ", bboxes, "\n", pred_conf, "\n", pred_labels)

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

    # Save generated image with detections
    output_path = f"test_results/{name}"
    cv2.imwrite(output_path, im)

    # # Display image
    # cv2.imshow("Prediction", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."

    root_subgraph = (
        graph.get_root_subgraph()
    )  # Retrieves the root subgraph of the input 'graph'
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."

    if root_subgraph.is_leaf:
        return (
            []
        )  # If it is a leaf, it means there are no child subgraphs, so the function returns an empty list

    child_subgraphs = (
        root_subgraph.toposort_child_subgraph()
    )  # Retrieves a list of child subgraphs of the 'root_subgraph' in topological order
    assert child_subgraphs is not None and len(child_subgraphs) > 0

    return [
        # List comprehension that filters the child_subgraphs list to include only those subgraphs that represent DPUs
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def preprocess_one_image_fn(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (W, H))
    image = image.astype(np.float32) / 255.0
    # Standardize the image
    image -= MEANS
    image /= STD
    return image


def main(argv):

    images_path = argv[2]

    for file in os.listdir(images_path):
        image_path = os.path.join(images_path, file)

        image = preprocess_one_image_fn(image_path)

        g = xir.Graph.deserialize(argv[1])  # Deserialize the DPU graph
        subgraphs = get_child_subgraph_dpu(g)  # Extract DPU subgraphs from the graph
        assert len(subgraphs) == 1  # only one DPU kernel

        time_start = time.time()

        """Creates DPU runner, associated with the DPU subgraph."""
        dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

        runYolo(dpu_runner, image, image_path, name=file)

        del dpu_runner

    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = 1

    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage : python3 dpu_inference.py <xmodel_file> <images_path>")
    else:
        main(sys.argv)
