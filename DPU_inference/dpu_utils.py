import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process_preds(preds, S, SCALE, anchor_boxes):
    """
    Converts the target tensor back to bounding boxes and labels.

    Parameters:
    - preds (numpy.ndarray): The predicted tensor.
    - S (list): The sizes of the grids.
    - SCALE (list): The scale factors.
    - anchor_boxes (list): List of anchor boxes.

    Returns:
    - list: List of processed predictions.
    """
    new_preds = []

    for i, pred in enumerate(preds):  # multiple targets
        # Apply sigmoid to certain components
        pred[..., 0:1] = sigmoid(pred[..., 0:1])
        pred[..., 1:3] = sigmoid(pred[..., 1:3])
        pred[..., 3:5] = (sigmoid(pred[..., 3:5])*2)**3
        

        cx = np.tile(np.arange(S[i]), (S[i], 1)).T  # Grid for x center offsets
        cy = np.tile(np.arange(S[i]), (S[i], 1))  # Grid for y center offsets

        # converint into the shape like (13, 13, 1, 1) so that boardacasting can be done
        cx = cx[..., np.newaxis, np.newaxis]
        cy = cy[..., np.newaxis, np.newaxis]

        pred[..., 1:2] += cx  # boardcastin across all the values
        pred[..., 2:3] += cy

        pred[..., 3:5] *= anchor_boxes[i]

        # Scale the predictions
        pred[..., 1:3] *= SCALE[i]

        pred[..., 3:5] *= SCALE[i]

        new_preds.append(pred)

    return new_preds


def non_max_suppression(prediction, iou_threshold=0.4, size=416):
    """Perform non-maximum suppression to remove overlapping bounding boxes based on scores.
    Args:
        prediction: Numpy array of shape (N, 5), where N is the number of bounding boxes,
                    and each bounding box is represented as [score, cx, cy, w, h].
        iou_threshold: IoU threshold for suppression.

    Returns:
        indices: Indices of the boxes that should be kept after non-maximum suppression.
    """
    # Calculate (x1, y1, x2, y2) from (center_x, center_y, width, height)
    cx, cy, w, h = (
        prediction[:, 1],
        prediction[:, 2],
        prediction[:, 3],
        prediction[:, 4],
    )

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = np.stack((x1, y1, x2, y2), axis=1)

    # creating a deep copy of converted boxes
    converted_bbox = np.copy(boxes)

    scores = prediction[:, 0]

    # Sort boxes by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    boxes = boxes[sorted_indices]

    selected_indices = []

    while len(boxes) > 0:
        # Pick the box with the highest score
        max_box = boxes[0]
        selected_indices.append(sorted_indices[0])

        if len(boxes) == 1:
            break

        # Calculate IoU with the rest of the boxes
        intersection_x1 = np.maximum(max_box[0], boxes[1:, 0])
        intersection_y1 = np.maximum(max_box[1], boxes[1:, 1])
        intersection_x2 = np.minimum(max_box[2], boxes[1:, 2])
        intersection_y2 = np.minimum(max_box[3], boxes[1:, 3])

        intersection_area = np.maximum(
            0, intersection_x2 - intersection_x1
        ) * np.maximum(0, intersection_y2 - intersection_y1)
        box_area = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        rest_areas = (boxes[1:, 2] - boxes[1:, 0]) * (boxes[1:, 3] - boxes[1:, 1])

        iou = intersection_area / (box_area + rest_areas - intersection_area)

        # Filter out boxes with IoU greater than the threshold
        filtered_indices = np.where(iou <= iou_threshold)[0]

        # Update boxes and sorted_indices
        boxes = boxes[filtered_indices + 1]
        sorted_indices = sorted_indices[filtered_indices + 1]

    sel_bboxes = np.clip(
        converted_bbox[selected_indices], 0, size
    )  # in cordinate format
    sel_scores = prediction[selected_indices][..., 0]
    sel_labels = np.argmax(prediction[selected_indices][..., 5:], axis=-1)

    return sel_bboxes, sel_scores, sel_labels
