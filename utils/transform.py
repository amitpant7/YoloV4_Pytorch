import torch
from torchvision.transforms import v2
from .dataset import MyCustomTransformatioms, FinalTranform
from torchvision import tv_tensors

transformations = v2.Compose(
    [
        v2.RandomPhotometricDistort(p=0.3),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomZoomOut(
            p=0.2, side_range=(1.0, 1.3), fill={tv_tensors.Image: (128, 128, 128)}
        ),
        #     v2.RandomIoUCrop(min_scale = 0.9, max_scale = 1, max_aspect_ratio=1.25, min_aspect_ratio=0.75),
        # #     v2.Resize((416,416), antialias=True),
        v2.RandomPerspective(distortion_scale=0.2, p=0.1),
        v2.RandomRotation(degrees=20),
        v2.RandomResizedCrop(size=(416, 416), scale=(0.9, 1), antialias=True),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.SanitizeBoundingBoxes(),
        FinalTranform(),
    ]
)

train_transform = MyCustomTransformatioms(transformations)

val_transform = MyCustomTransformatioms(
    v2.Compose(
        v2.RandomResizedCrop(size=(416, 416), scale=(0.9, 1), antialias=True),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        FinalTranform(),
    )
)


rev_transform = v2.Compose(
    [
        v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        v2.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)
