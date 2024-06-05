import torchvision
import torch
from tqdm.auto import tqdm
from config import *
from .utils import convert_to_corners, visualize_bb


def process_preds(preds, S=S, SCALE=SCALE, anchor_boxes=ANCHOR_BOXES):
    """
    Converts the target tensor back to bounding boxes and labels.

    Parameters:
    - ground_truth (torch.Tensor): The ground truth tensor.
    - S (int, optional): The size of the grid. Default is 13.
    - SCALE (int, optional): The scale factor. Default is 32.
    - anchor_boxes (list, optional): List of anchor boxes. Default is None.

    Returns:
    - tuple: (bbox, labels) where bbox are the bounding boxes and labels are the object labels.
    """
    # Each list element will have reversed targets, i.e ground truth bb
    # Just for verifying all the targets are properly build, if they can be reversed then good.
    new_preds = []
    for i, pred in enumerate(preds):  # multiple targets
        bboxes = []
        labels = []
        pred = pred.to(device)
        #         print('Before:', pred[0, 12, 12, 1])

        pred[..., 0:1] = torch.sigmoid(pred[..., 0:1])
        pred[..., 1:3] = torch.sigmoid(pred[..., 1:3])
        pred[..., 3:5] = torch.exp(pred[..., 3:5])

        cx = cy = torch.tensor([i for i in range(S[i])], device=device)
        pred = pred.permute(0, 3, 4, 2, 1)
        pred[..., 1:2, :, :] += cx
        pred = pred.permute(0, 1, 2, 4, 3)
        pred[..., 2:3, :, :] += cy
        pred = pred.permute((0, 3, 4, 1, 2))
        pred[..., 1:3] *= SCALE[i]
        pred[..., 3:5] *= anchor_boxes[i].to(device)
        pred[..., 3:5] = pred[..., 3:5] * SCALE[i]
        new_preds.append(pred)
    #         print('After:', pred[0, 12, 12, 1])
    return new_preds


def non_max_suppression(boxes, scores, io_threshold):
    """
    Perform non-maximum suppression to eliminate redundant bounding boxes based on their scores.

    Args:
        boxes (Tensor): Tensor of shape (N, 4) containing bounding boxes in the format (x_center, y_center, width, height).
        scores (Tensor): Tensor of shape (N,) containing confidence scores for each bounding box.
        threshold (float): Threshold value for suppressing overlapping boxes.

    Returns:
        Tensor: Indices of the selected bounding boxes after NMS.
    """
    # Convert bounding boxes to [x_min, y_min, x_max, y_max] format
    boxes = convert_to_corners(boxes)
    #     print(boxes)

    # Apply torchvision.ops.nms
    keep = torchvision.ops.nms(boxes, scores, io_threshold)

    return keep
