import torch

NO_OF_ANCHOR_BOX = N = 3

S = [13, 26, 52]  # Three output prediction Scales of Yolov3

NO_OF_CLASS = C = 20
HEIGHT = H = 416
WIDTH = W = 416
SCALE = [32, 16, 8]


DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = batch_size = 16


ANCHOR_BOXES = A = torch.tensor(
    [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]
) * torch.tensor(S).view(
    -1, 1, 1
)  # sale up in range 0-S[i], accordingly


CLASS_ENCODING = class_encoding = {
    "person": 0,
    "bird": 1,
    "cat": 2,
    "cow": 3,
    "dog": 4,
    "horse": 5,
    "sheep": 6,
    "aeroplane": 7,
    "bicycle": 8,
    "boat": 9,
    "bus": 10,
    "car": 11,
    "motorbike": 12,
    "train": 13,
    "bottle": 14,
    "chair": 15,
    "diningtable": 16,
    "pottedplant": 17,
    "sofa": 18,
    "tvmonitor": 19,
}

class_decoding = {v: k for k, v in class_encoding.items()}

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
