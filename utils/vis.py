from config import *
from .postprocess import process_preds, non_max_suppression
from .utils import visualize_bb


def visualize_outputs(
    indices, model, dataset, device=DEVICE, thres=0.6, iou_threshold=0.5
):
    """
    Visualizes the output predictions of the model on a set of images from the dataset.

    Args:
        indices (list of int): List of indices of the images to visualize.
        model (torch.nn.Module): The trained model to use for predictions.
        dataset (torch.utils.data.Dataset): The dataset containing the images and targets.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        thres (float, optional): The threshold for objectness score to filter predictions. Defaults to 0.9.

    Returns:
        None
    """
    images_with_bb = []

    for index in indices:
        # Load the image and target from the dataset
        image, target = dataset[index]
        image = image.to(device)
        model = model.to(device)

        # Set the model to evaluation mode
        model.eval()

        # Get predictions from the model
        preds = model(image.unsqueeze(0))

        # Process the predictions
        preds = process_preds(preds)

        # Lets concatinate all 3 preds into single tensor and
        # Filter predictions based on the threshold
        preds = torch.cat([pred[pred[..., 0] > thres] for pred in preds], dim=0)

        bboxes = preds[..., 1:5]
        scores = torch.flatten(preds[..., 0])
        _, ind = torch.max(preds[..., 5:], dim=-1)
        classes = torch.flatten(ind)

        # Apply non-max suppression to get the best bounding boxes
        best_boxes = non_max_suppression(bboxes, scores, io_threshold=iou_threshold)

        filtered_bbox = bboxes[best_boxes]
        filtered_classes = classes[best_boxes]
        print(filtered_classes)

        if filtered_classes.size(0) > 0:
            sample = {
                "image": image.detach().cpu(),
                "bbox": filtered_bbox.detach().cpu(),
                "labels": filtered_classes.detach().cpu(),
            }

            images_with_bb.append(sample)

    visualize_bb(images_with_bb)
