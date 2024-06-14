# mean average precision calculation
import numpy as np
import torch
from tqdm.auto import tqdm
from torchvision import ops
from torcheval.metrics import AUC
from torchvision.datasets import VOCDetection

from config import *
from utils.postprocess import process_preds, non_max_suppression
from utils.transform import *


def mean_average_precision(
    model,
    max_samples=1000,
    dataset=val_data,
    iou_threshold_for_corr_pred=0.5,
    nms_thre=0.5,
):
    ep = 1e-6
    # dateset- ubatched entire val data
    pr_matrix = torch.zeros((9, C, 2))

    val_size = len(dataset)
    val_indices = np.random.choice(val_size, max_samples, replace=False)

    for i in range(1, 10):
        conf = i / 10

        confusion_matrix = torch.zeros(
            NO_OF_CLASS, 3
        )  # corr_preds, total_preds, actual_count for every class

        count = 0
        for index in tqdm(val_indices):
            img, targets = dataset[index]
            gt_bboxes, gt_labels = targets

            if gt_bboxes.size(0) == 0:
                continue

            # predicting outputs
            predictions = model(img.unsqueeze(0))
            predictions = process_preds(predictions)

            filtered_outputs = []
            for output in predictions:
                filtered_outputs.append(output[output[..., 0] >= conf])

            # concatinating all outputs in shape (total, 25)
            all_outputs = torch.cat(filtered_outputs, dim=0)

            scores, bboxes, classes = (
                all_outputs[:, 0],
                all_outputs[:, 1:5],
                all_outputs[..., 5:],
            )

            # performing non max supression
            keep = non_max_suppression(bboxes, scores, nms_thre)
            bboxes = bboxes[keep]
            _, classes = torch.max(classes[keep], dim=-1)

            for c in range(C):
                pred_mask = classes == c
                gt_mask = gt_labels == c

                if not pred_mask.any() or not gt_mask.any():
                    continue

                ious = ops.box_iou(bboxes[pred_mask], gt_bboxes[gt_mask])

                best_ious, best_idxs = ious.max(dim=1)

                corr_preds = (best_ious > iou_threshold_for_corr_pred).sum().item()
                total_preds = pred_mask.sum().item()
                actual_count = gt_mask.sum().item()

                confusion_matrix[c] += torch.tensor(
                    [corr_preds, total_preds, actual_count]
                )

        precision = confusion_matrix[:, 0] / (confusion_matrix[:, 1] + ep)
        recall = confusion_matrix[:, 0] / (confusion_matrix[:, 2] + ep)
        pr_matrix[i - 1] = torch.cat((precision.view(-1, 1), recall.view(-1, 1)), dim=1)
    #         print(pr_matrix[i - 1])
    pr_matrix = pr_matrix.permute(1, 0, 2)  # now shape class, all pr values
    # lets calculate the mean precision
    print(torch.isnan(pr_matrix).any())  # Should be False
    print(torch.isinf(pr_matrix).any())
    print(pr_matrix)
    metric = AUC(n_tasks=C)
    metric.update(pr_matrix[..., 0], pr_matrix[..., 1])
    average_precision = metric.compute()

    return average_precision.nanmean()
