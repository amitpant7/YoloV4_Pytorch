import torch
import numpy as np
from config import *
from .transform import rev_transform


def convert_to_corners(bboxes):
    """
    Convert bounding boxes from center format (center_x, center_y, width, height)
    to corner format (x1, y1, x2, y2).

    Parameters
    ----------
    bboxes : torch.Tensor
        Tensor of shape (B, 4) where B is the batch size.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B, 4) in corner format.
    """

    cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def intersection_over_union(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) between bounding boxes, expects batch dimension

    Parameters
    ----------
    bb1 : torch.Tensor
        Tensor of shape (B, 4) in center format (center_x, center_y, width, height).
    bb2 : torch.Tensor
        Tensor of shape (B, 4) in center format (center_x, center_y, width, height).

    Returns
    -------
    torch.Tensor
        Tensor of shape (B,) containing IoU for each pair of bounding boxes.
    """
    # Convert center-width-height format to top-left and bottom-right format
    bboxes1 = convert_to_corners(bb1)
    bboxes2 = convert_to_corners(bb2)

    # Calculate the coordinates of the intersection rectangles
    x_left = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    y_top = torch.max(bboxes1[:, 1], bboxes2[:, 1])
    x_right = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    y_bottom = torch.min(bboxes1[:, 3], bboxes2[:, 3])

    # Calculate the intersection area
    intersection_width = torch.clamp(x_right - x_left, min=0)
    intersection_height = torch.clamp(y_bottom - y_top, min=0)
    intersection_area = intersection_width * intersection_height

    # Calculate the area of each bounding box
    bb1_area = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    bb2_area = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    # Calculate the IoU
    iou = intersection_area / (bb1_area + bb2_area - intersection_area)

    return iou


import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes


def show(imgs):
    """
    Displays a list of images in a grid format.

    Args:
        imgs (list of torch.Tensor): List of images to be displayed.

    Returns:
        None
    """
    total_images = len(imgs)
    num_rows = (total_images + 1) // 2  # Calculate the number of rows
    fig, axs = plt.subplots(nrows=num_rows, ncols=2, squeeze=False, figsize=(12, 12))

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        row_idx = i // 2
        col_idx = i % 2
        axs[row_idx, col_idx].imshow(np.asarray(img))
        axs[row_idx, col_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


def visualize_bb(samples):
    """
    Visualizes bounding boxes on a list of images.

    Args:
        samples (list of dict): List of samples, each containing an image, bounding boxes, and labels.

    Returns:
        None
    """
    colors = COLORS
    images = []
    for sample in samples:
        img = sample["image"].to("cpu")
        img = rev_transform(img)
        img = (img * 255).to(torch.uint8)
        bboxes = sample["bbox"].to("cpu").numpy()
        labels = sample["labels"].to("cpu")

        _, height, width = img.size()

        corr_bboxes = []
        for bbox in bboxes:
            x, y = bbox[0], bbox[1]  # Center of the bounding box
            box_width, box_height = bbox[2], bbox[3]

            # Calculate the top-left and bottom-right corners of the rectangle
            x1 = int(x - box_width / 2)
            y1 = int(y - box_height / 2)
            x2 = int(x + box_width / 2)
            y2 = int(y + box_height / 2)

            corr_bboxes.append([x1, y1, x2, y2])

        corr_bboxes = torch.tensor(
            corr_bboxes
        )  # Convert to tensor for draw_bounding_boxes
        img_with_bbox = draw_bounding_boxes(
            img, corr_bboxes, colors=[colors[label] for label in labels], width=3
        )
        images.append(img_with_bbox)

    show(images)


def copy_wts(model, source_wts):
    """Transfer weights from pretrained model

    Args:
        model (_type_): Your YOLO model
        path (str, optional): path of state dictionary for copying wts.

    Returns:
        model: model with updated wts.
    """

    count = 0

    wts = model.state_dict()
    org_wt = source_wts
    org_key_list = list(org_wt.keys())

    for key1 in org_key_list:
        for key2 in wts.keys():
            if org_wt[key1].shape == wts[key2].shape:
                count += 1
                wts[key2] = org_wt[key1]  # copy model wts
                break

    print("Total Layers Matched:", count)

    model.load_state_dict(wts)

    return model


def check_model_accuracy(all_preds, all_targets, thres=0.5):
    """
    all_preds: list of batches of list of tensors [[3 preds], ...]
    all_targets: list of batches of list of tensors [[3 targets], ...]
    thres: threshold for objectness score
    """
    with torch.no_grad()
        total_class, class_corr = 0, 0
        total_obj, obj_corr = 0, 0
        total_no_obj, no_obj_corr = 0, 0
        total_class_preds, correct_class_preds = 0, 0

        sig = torch.nn.Sigmoid()

        preds = []  # concatenated preds in format [pred1, pred2, pred3]
        targets = []
        for i in range(3):
            preds.append(torch.cat([x[i] for x in all_preds], dim=0))
            targets.append(torch.cat([x[i] for x in all_targets], dim=0))

        for i in range(len(preds)):
            obj = targets[i][..., 0] == 1  # mask
            no_obj = targets[i][..., 0] == 0

            preds[i][..., 0] = sig(preds[i][..., 0])

            # Classification Accuracy
            class_pred = torch.argmax(preds[i][obj][..., 5:], dim=-1)
            class_target = torch.argmax(targets[i][obj][..., 5:], dim=-1)
            class_corr += torch.sum(class_pred == class_target)
            total_class += torch.sum(obj)

            # Object detection recall and precision
            obj_corr += torch.sum(preds[i][obj][..., 0] > thres)
            total_obj += torch.sum(obj) + 1e-6  # to avoid divide by zero
            obj_preds = preds[i][..., 0] > thres
            correct_obj_preds = obj & obj_preds
            total_obj_preds = torch.sum(obj_preds) + 1e-6

            # No-object detection recall and precision
            no_obj_corr += torch.sum(preds[i][no_obj][..., 0] < thres)
            total_no_obj += torch.sum(no_obj)
            no_obj_preds = preds[i][..., 0] < thres
            correct_no_obj_preds = no_obj & no_obj_preds
            total_no_obj_preds = torch.sum(no_obj_preds)

        class_score = (100 * class_corr / total_class).item()
        # Recall calculations
        obj_recall = (100 * obj_corr / total_obj).item()
        no_obj_recall = (100 * no_obj_corr / total_no_obj).item()

        # Precision calculations
        obj_precision = (100 * correct_obj_preds.sum() / total_obj_preds).item()
        no_obj_precision = (100 * correct_no_obj_preds.sum() / total_no_obj_preds).item()

    print("Class Score (Accuracy): {:.2f}%".format(class_score))
    print("Object Score (Recall): {:.2f}%".format(obj_recall))
    print("Object Score (Precision): {:.2f}%".format(obj_precision))
    print("No-object Score (Recall): {:.2f}%".format(no_obj_recall))
    print("No-object Score (Precision): {:.2f}%".format(no_obj_precision))
