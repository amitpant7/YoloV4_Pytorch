import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torchvision.ops import complete_box_iou_loss
import torch.nn as nn

from utils.utils import convert_to_corners
from config import C, S, DEVICE, ANCHOR_BOXES


def label_smoothing(labels, smoothing_factor=0.1):
    """
    Apply label smoothing to one-hot encoded labels.

    Parameters:
    - one_hot_labels (np.ndarray): One-hot encoded labels with shape (num_samples, num_classes).
    - smoothing_factor (float): The smoothing factor (usually small, e.g., 0.1).

    Returns:
    - np.ndarray: Smoothed labels with the same shape as `one_hot_labels`.
    """
    if smoothing_factor < 0.0 or smoothing_factor > 1.0:
        raise ValueError("Smoothing factor should be in the range [0, 1].")

    num_classes = labels.shape[1]
    # Smooth labels by distributing smoothing_factor across all classes
    smoothed_labels = (1 - smoothing_factor) * labels + (smoothing_factor / num_classes)

    return smoothed_labels


def ciou(pred_box, gt_box):
    pred_box = convert_to_corners(pred_box)
    gt_box = convert_to_corners(gt_box)
    pred_box = torch.clamp(pred_box, min=0)

    loss = complete_box_iou_loss(pred_box, gt_box)
    ious = 1 - loss.detach()

    return loss.nanmean(), ious


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.ce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        pred_prob = torch.sigmoid(pred)

        # pt pt=true×pred_prob+(1−true)×(1−pred_prob).
        pt = target * pred_prob + (1 - target) * (1 - pred_prob)
        coeff = (1 - pt) ** self.gamma

        focal_loss = ce_loss * coeff
        return focal_loss.nanmean()


class YoloV4_Loss(torch.nn.Module):
    """
    YOLOv3 Loss Function

    This class implements the loss function for the YOLOv3 object detection model.
    It includes components for objectness, bounding box regression, and class probabilities.

    Attributes:
        lambda_no_obj (torch.Tensor): Weight for no-object loss.
        lambda_obj (torch.Tensor): Weight for object loss.
        lambda_class (torch.Tensor): Weight for class probability loss.
        lambda_bb_cord (torch.Tensor): Weight for bounding box coordinate loss.
        C (int): Number of classes.
        S (list): All Scales
        binary_loss (torch.nn.Module): Binary cross-entropy loss with logits.
        logistic_loss (torch.nn.Module): Cross-entropy loss for class probabilities.
        regression_loss (torch.nn.Module): Mean squared error loss for bounding box regression.
    """

    def __init__(self, C=C, S=S, device=DEVICE, anchor_boxes=ANCHOR_BOXES):
        """
        Initializes the YOLOv3 loss function.

        Args:
            C (int): Number of classes.
            S (list): Scales.
            device (str, optional): Device to place the tensors on. Defaults to 'cpu'.
        """
        super().__init__()
        self.device = device
        self.lambda_no_obj = torch.tensor(1.0, device=device)
        self.lambda_obj = torch.tensor(8.0, device=device)
        self.lambda_class = torch.tensor(10.0, device=device)  # 7 without focal
        self.lambda_bb = torch.tensor(1.5, device=device)

        self.C = C
        self.S = S
        self.A = anchor_boxes

        # Loss functions
        # self.binary_loss = BCEWithLogitsLoss()  # Binary cross-entropy with logits
        self.logistic_loss = CrossEntropyLoss(
            label_smoothing=0.2
        )  # Cross-entropy loss for class probabilities

        self.regression_loss = MSELoss()

        self.focal = FocalLoss(gamma=2)

    def forward(self, preds, ground_truths):
        """
        Computes the YOLOv3 loss.

        Args:
            preds (list[torch.Tensor]): Predictions from the model for different scales. Shape (B, S, S, A*(5+C)).
            ground_truths (list[torch.Tensor]): Ground truth labels for different scales. Shape (B, S, S, A*(5+C)).

        Returns:
            torch.Tensor: Total loss.
        """
        losses = []

        for i in range(len(self.S)):
            pred = preds[i]
            ground_truth = ground_truths[i]

            # Identify object and no-object cells
            obj = ground_truth[..., 0] == 1
            no_obj = ground_truth[..., 0] == 0

            # TODO
            # in dataset prep don't do log and divide by anchors, remove processing for gt to cx,cy as no longer need in localization

            if torch.sum(obj) > 0:
                pred[..., 1:3] = torch.sigmoid(pred[..., 1:3])
                pred[..., 3:5] = (torch.sigmoid(pred[..., 3:5]) * 2) ** 3
                ground_truth[..., 3:5] = torch.exp(
                    ground_truth[..., 3:5]
                )  # log used in gt

                #             reg_loss = self.regression_loss(pred[obj][1:5], ground_truth[obj][1:5])

                cx = cy = torch.tensor([i for i in range(S[i])]).to(self.device)
                pred = pred.permute(0, 3, 4, 2, 1)
                pred[..., 1:2, :, :] += cx

                pred = pred.permute(0, 1, 2, 4, 3)
                pred[..., 2:3, :, :] += cy
                pred = pred.permute((0, 3, 4, 1, 2))
                pred[..., 3:5] *= self.A[i].to(self.device)

                ground_truth = ground_truth.permute(0, 3, 4, 2, 1)
                ground_truth[..., 1:2, :, :] += cx

                ground_truth = ground_truth.permute(0, 1, 2, 4, 3)
                ground_truth[..., 2:3, :, :] += cy
                ground_truth = ground_truth.permute((0, 3, 4, 1, 2))
                ground_truth[..., 3:5] *= self.A[i].to(self.device)

                # Bounding box loss
                pred_bb = pred[obj][..., 1:5] * SCALE[i]
                gt_bb = ground_truth[obj][..., 1:5] * SCALE[i]

                bb_cord_loss, ious = ciou(pred_bb, gt_bb)
                ious = ious.clamp(min=0.4, max=1.0)

                # use focal loss insted of object, no object loss
                obj_loss = self.focal(
                    pred[obj][..., 0], ground_truth[obj][..., 0] * ious
                )
                noobj_loss = self.focal(
                    pred[no_obj][..., 0], ground_truth[no_obj][..., 0]
                )

                # class loss

                # class_loss = self.logistic_loss(pred[obj][..., 5:], ground_truth[obj][..., 5:])
                smoothed_class = label_smoothing(
                    ground_truth[obj][..., 5:], smoothing_factor=0.15
                )
                class_loss = self.focal(pred[obj][..., 5:], smoothed_class)

                # Total loss calculation with weighted components
                loss = (
                    self.lambda_bb * bb_cord_loss
                    + self.lambda_obj * obj_loss
                    + self.lambda_no_obj * noobj_loss
                    + self.lambda_class * class_loss
                )

                losses.append(loss)

            else:
                noobj_loss = self.focal(
                    pred[no_obj][..., 0], ground_truth[no_obj][..., 0]
                )
                # Total loss calculation with weighted components
                loss = self.lambda_no_obj * noobj_loss

                losses.append(loss)

        #             print("Loss Values", bb_cord_loss.item(),obj_loss.item(), noobj_loss.item(), class_loss.item())
        total_loss = torch.stack(losses).sum()

        return total_loss
