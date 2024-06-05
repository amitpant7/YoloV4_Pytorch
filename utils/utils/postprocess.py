import torchvision
import torch
from torchvision import ops
from torcheval.metrics import AUC
from tqdm.auto import tqdm
from config import *
from .dataset import inverse_target
from .utils import convert_to_corners


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


# depends on non_max_supression_implementation and post_processing
# iou_thres_for_corr_predn-.
def mean_average_precision(
    predictions, targets, iou_thres_nms=0.4, iou_thres_for_corr_predn=0.5, C=C
):
    """Calculates Mean avg precision for a single batch, to calculate for all batch collect prediction
    and targets in a tensor and pass it here

    Args:
        predictions (_type_): Model outputs in the tensor format (B, S,S,N,C+5)
        targets (_type_): _description_
        data (_type_): Custom Dataset instance
        iou_thres_nms (float, optional): Threshold for IOU in non max supression. Defaults to 0.5.
        iou_thres_for_corr_predn (float, optional):  Min iou with ground bb to consider it as correct prediction. Defaults to 0.4.

    """

    ep = 1e-6

    # getting back pixel values:
    ground_truths = targets
    processed_preds = process_preds(predictions)
    pr_matrix = torch.empty(
        9, C, 2
    )  # Precision and recall values at 9 different levels of threh(confidance score)

    for thres in range(1, 10, 1):

        conf_thres = thres / 10
        local_pr_matrix = torch.zeros(
            C, 3
        )  # Corr_pred, total_preds, ground_truth for every class

        for i in range(processed_preds[0].size(0)):  # looping over entire batch

            # processing the preds to make it suitable
            preds = [pred[i] for pred in processed_preds]

            obj = [pred[..., 0] > conf_thres for pred in preds]

            bboxes = torch.cat(
                (
                    preds[0][obj[0]][..., 1:5].view(-1, 4),
                    preds[1][obj[1]][..., 1:5].view(-1, 4),
                    preds[2][obj[2]][..., 1:5].view(-1, 4),
                ),
                dim=0,
            )

            scores = torch.cat(
                (
                    preds[0][obj[0]][..., 0].view(-1),
                    preds[1][obj[1]][..., 0].view(-1),
                    preds[2][obj[2]][..., 0].view(-1),
                ),
                dim=0,
            )

            _, ind0 = torch.max(preds[0][obj[0]][..., 5:], dim=-1)
            _, ind1 = torch.max(preds[1][obj[1]][..., 5:], dim=-1)
            _, ind2 = torch.max(preds[2][obj[2]][..., 5:], dim=-1)
            classes = torch.cat((ind0, ind1, ind2), dim=0)

            if bboxes.size(0) == 0:
                continue

            # nms to supress overlapping boxes
            keep = non_max_suppression(bboxes, scores, iou_thres_nms)
            bboxes = bboxes[keep]
            classes = classes[keep]

            gt = [gt[i].detach().clone().unsqueeze(0) for gt in ground_truths]
            #         print(filtered_bbox[filtered_classes==0])
            gt_bboxes, labels = inverse_target(gt)  # inverse_target expects batched
            gt_bboxes, labels = (
                gt_bboxes[0],
                labels[0],
            )  # the inverse_target returns list of 3 elements, targets for each scale, but one gt is enough.

            if gt_bboxes.size(0) == 0:
                continue

            #         print(gt_bboxes, labels)
            # matche the one bbox among the predicted boxes with the ground thruth box that gives higesht iou.
            tracker = torch.zeros(
                labels.size(0), dtype=torch.bool
            )  # to keep track of matched boxes

            # go throuch each class, count corr preds
            # Corr preds: match class predn with ground class, match predn boxes to gt bbox.
            for c in range(C):
                pred_mask = classes == c
                gt_mask = labels == c

                if not pred_mask.any() or not gt_mask.any():
                    continue

                ious = ops.box_iou(bboxes[pred_mask], gt_bboxes[gt_mask])

                best_ious, best_idxs = ious.max(dim=1)

                corr_preds = (best_ious > iou_thres_for_corr_predn).sum().item()
                total_preds = pred_mask.sum().item()
                actual_count = gt_mask.sum().item()

                local_pr_matrix[c] += torch.tensor(
                    [corr_preds, total_preds, actual_count]
                )

        #         print(local_pr_matrix)
        precision, recall = local_pr_matrix[:, 0] / (
            local_pr_matrix[:, 1] + ep
        ), local_pr_matrix[:, 0] / (
            local_pr_matrix[:, 2] + ep
        )  # pr at a certain threshold c
        pr_matrix[thres - 1] = torch.cat(
            (precision.view(-1, 1), recall.view(-1, 1)), dim=1
        )

    pr_matrix = pr_matrix.permute(1, 0, 2)  # now shape class, all pr values

    # lets calculate the mean precision
    print(pr_matrix.shape)
    metric = AUC(n_tasks=C)
    metric.update(pr_matrix[..., 0], pr_matrix[..., 1])
    average_precision = metric.compute()

    return average_precision.mean()
