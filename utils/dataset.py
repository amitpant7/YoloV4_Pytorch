import torch
from torchvision import datasets
from torchvision import tv_tensors
from ..config import *


# lets write custom tranform to transform the targets in appropriate format.


class MyCustomTransformatioms(torch.nn.Module):
    # pase the dictionary format of targets in pascal voc and create bboxes and labels , from it from it
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, img, data):
        img = tv_tensors.Image(img)
        labels = []
        bboxes = []

        class_encoding = CLASS_ENCODING

        annotation = data["annotation"]
        objects = annotation["object"]

        for obj in objects:
            label = obj["name"]
            bbox = obj["bndbox"]
            xmin = int(bbox["xmin"])
            ymin = int(bbox["ymin"])
            xmax = int(bbox["xmax"])
            ymax = int(bbox["ymax"])

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            width = xmax - xmin
            height = ymax - ymin

            labels.append(class_encoding[label])
            bboxes += [[x_center, y_center, width, height]]

        bboxes = tv_tensors.BoundingBoxes(
            bboxes, format="CXCYWH", canvas_size=img.shape[-2:]
        )

        sample = {"image": img, "labels": torch.tensor(labels), "bboxes": bboxes}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class FinalTranform(torch.nn.Module):
    # Retruns target in the shape [S, S, N, C+5] for every Scale,
    # So a tesor represtnation of target for all anchor boxes and all scale values .

    def __init__(self):
        super().__init__()

    def forward(self, sample):
        image = sample["image"]
        bboxes = sample["bboxes"]
        labels = sample["labels"]

        # building targets
        targets = []

        # for every scale[13,26,52]:

        for i in range(len(S)):
            to_exclude = []  # we won't assign same anchor box multiple times.

            target = torch.zeros(S[i], S[i], N, 1 + 4 + C)  # S*S*N, 1+4+C

            for bbox, label in zip(bboxes, labels):
                cx, cy = bbox[0] / SCALE[i], bbox[1] / SCALE[i]  # Float values
                pos = (int(cx), int(cy))
                bx, by = cx - int(cx), cy - int(cy)
                box_width, box_height = bbox[2] / SCALE[i], bbox[3] / SCALE[i]

                assigned_anchor_box, ignore_indices = match_anchor_box(
                    box_width, box_height, i, to_exclude
                )

                if assigned_anchor_box is None:
                    continue

                anchor_box = ANCHOR_BOXES[i][assigned_anchor_box]

                bw_by_Pw, bh_by_ph = (
                    box_width / anchor_box[0],
                    box_height / anchor_box[1],
                )

                epsilon = 1e-6

                target[pos[0], pos[1], assigned_anchor_box, 0:5] = torch.tensor(
                    [
                        1,
                        bx,
                        by,
                        torch.log(bw_by_Pw + epsilon),
                        torch.log(bh_by_ph + epsilon),
                    ]
                )
                target[pos[0], pos[1], assigned_anchor_box, 5 + int(label)] = 1

                to_exclude.append(assigned_anchor_box)

                try:
                    for value in ignore_indices:
                        target[pos[0], pos[1], value.item(), 0] = -1
                except:
                    pass

            targets.append(target)

        return image, targets


def match_anchor_box(
    bbox_w,
    bbox_h,
    i,
    to_exclude=[],
):
    """
    Matches the bounding box to the closest anchor box.

    Parameters:
    - bbox_w (float): The width of the bounding box.
    - bbox_h (float): The height of the bounding box.
    - to_exclude (list): List of anchor boxes to exclude.

    Returns:
    - int: Index of the matched anchor box.
    """
    ignore = 0.5
    anchor_boxes = ANCHOR_BOXES[i]
    iou = []
    for i, box in enumerate(anchor_boxes):
        if i in to_exclude:
            iou.append(0)
            continue
        intersection_width = min(box[0], bbox_w)  # Scale up as h, w in range 0-13
        intersection_height = min(box[1], bbox_h)
        I = intersection_width * intersection_height
        IOU = I / (bbox_w * bbox_h + box[0] * box[1] - I)
        iou.append(IOU)

    iou = torch.tensor(iou)
    best = torch.argmax(iou, dim=0).item()

    # I want to not assign anchor if the IOU is below this.

    if iou[best] < 0.2:
        best = None

    # Ignore anchors if they have high IOU but are not the best match
    ignore_indices = torch.nonzero((iou > ignore) & (iou != iou[best])).squeeze()

    return best, ignore_indices


def inverse_target(ground_truths, S=S, SCALE=SCALE, anchor_boxes=ANCHOR_BOXES):
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
    all_bboxes = []
    all_labels = (
        []
    )  # Just for verifying all the targets are properly build, if they can be reversed then good.

    for i, ground_truth in enumerate(ground_truths):  # multiple targets
        bboxes = []
        labels = []
        ground_truth = ground_truth.to(device)
        cx = cy = torch.tensor([i for i in range(S[i])], device=device)

        ground_truth = ground_truth.permute(0, 3, 4, 2, 1)
        ground_truth[..., 1:2, :, :] += cx
        ground_truth = ground_truth.permute(0, 1, 2, 4, 3)
        ground_truth[..., 2:3, :, :] += cy
        ground_truth = ground_truth.permute((0, 3, 4, 1, 2))

        ground_truth[..., 1:3] *= SCALE[i]
        ground_truth[..., 3:5] = torch.exp(ground_truth[..., 3:5])
        ground_truth[..., 3:5] *= anchor_boxes[i].to(device)
        ground_truth[..., 3:5] = ground_truth[..., 3:5] * SCALE[i]

        bbox = ground_truth[ground_truth[..., 0] == 1][..., 1:5]
        _, labels = torch.max(
            ground_truth[ground_truth[..., 0] == 1][..., 5:].view(-1, C), dim=-1
        )
        all_bboxes.append(bbox)
        all_labels.append(labels)

    return all_bboxes, all_labels
