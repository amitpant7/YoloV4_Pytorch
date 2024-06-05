import torch
from config import *
from torcheval.metrics import AUC
from torchvision import ops

from .postprocess import non_max_suppression, process_preds
from .dataset import inverse_target


def evaluate_model(model, dataloaders, device=DEVICE, phase="val"):
    """
    Evaluates a PyTorch model on the validation dataset.

    Args:
    - model (torch.nn.Module): The PyTorch model to be evaluated.
    - dataloaders (dict): A dictionary containing dataloaders for different phases (e.g., 'train', 'val').
    - dataset_sizes (dict): A dictionary containing the sizes of datasets for different phases.
    - batch_size (int): The batch size for evaluation.
    - device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to perform evaluation.

    Returns:
    - all_preds (torch.Tensor): Predictions made by the model, reshaped for evaluation.
    - all_targets (torch.Tensor): Ground truth labels, reshaped for evaluation.
    """

    model = model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for image, targets in dataloaders[phase]:
            image = image.to(device)
            preds = model(image)

            targets = [
                target.detach().to("cpu") for target in targets
            ]  # storing in cpu so no run out of mem
            all_preds.append([pred.detach().to("cpu") for pred in preds])
            all_targets.append(targets)

    preds = []  # concatenated preds in format [pred1, pred2, pred3]
    targets = []

    for i in range(3):
        preds.append(torch.cat([x[i] for x in all_preds], dim=0))
        targets.append(torch.cat([x[i] for x in all_targets], dim=0))

    # moving back tensors to gpu for faster caln
    preds = [pred.to(device) for pred in preds]
    targets = [target.to(device) for target in targets]

    map = mean_average_precision(preds, targets)
    print("The mean average precision on {phase}:", map.item())
    return map


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
