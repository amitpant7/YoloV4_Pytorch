import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torchvision.ops import complete_box_iou_loss

from utils.utils import convert_to_corners
from config import C, S, DEVICE


def ciou(pred_box, gt_box):
    pred_box = convert_to_corners(pred_box)
    gt_box = convert_to_corners(gt_box)
    loss = complete_box_iou_loss(pred_box, gt_box, reduction='mean')
    # ious = 1 - loss

    # if torch.isnan(loss).all():
    #     # Handle the case where all elements are nan
    #     print("All elements are nan.")

    return loss.nanmean(), 0


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

    def __init__(self, C=C, S=S, device=DEVICE):
        """
        Initializes the YOLOv3 loss function.

        Args:
            C (int): Number of classes.
            S (list): Scales.
            device (str, optional): Device to place the tensors on. Defaults to 'cpu'.
        """
        super().__init__()
        self.lambda_no_obj = torch.tensor(0.5, device=device)
        self.lambda_obj = torch.tensor(1.0, device=device)
        self.lambda_class = torch.tensor(1.0, device=device)  # 3,5 in prev
        self.lambda_bb_cord = torch.tensor(2.0, device=device)
        self.C = C
        self.S = S

        # Loss functions
        self.binary_loss = BCEWithLogitsLoss()  # Binary cross-entropy with logits
        self.logistic_loss = CrossEntropyLoss(
            label_smoothing=0.1
        )  # Cross-entropy loss for class probabilities
        self.regression_loss = (
            MSELoss()
        )  # Mean squared error loss for bounding box regression

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

            #avoid loss calculation if there aren't any targets assigned
            is_zero = torch.all(ground_truth == 0)
            
            if is_zero:
                continue

            # Identify object and no-object cells
            obj = ground_truth[..., 0] == 1
            no_obj = ground_truth[..., 0] == 0

            # No-object loss
            no_obj_loss = self.binary_loss(
                pred[no_obj][..., 0], ground_truth[no_obj][..., 0]
            )

            # Bounding box loss
            pred_bb = torch.cat(
                (torch.sigmoid(pred[obj][..., 1:3]), pred[obj][..., 3:5]), dim=-1
            )
            gt_bb = ground_truth[obj][..., 1:5]

            bb_cord_loss, ious = ciou(pred_bb, gt_bb)

            # Object loss
            obj_loss = self.binary_loss(pred[obj][..., 0], ground_truth[obj][..., 0])

            # Class probability loss
            pred_prob = pred[obj][..., 5:]
            class_loss = self.logistic_loss(pred_prob, ground_truth[obj][..., 5:])

            # Total loss calculation with weighted components
            loss = (
                self.lambda_bb_cord * bb_cord_loss
                + self.lambda_no_obj * no_obj_loss
                + self.lambda_obj * obj_loss
                + self.lambda_class * class_loss
            )

            losses.append(loss)
        total_loss = torch.stack(losses).mean()

        print("Loss Values", bb_cord_loss.item(), no_obj_loss.item(), obj_loss.item(), class_loss.item(), total_loss.item())
        return total_loss
