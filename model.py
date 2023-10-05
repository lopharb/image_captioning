import numpy as np
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from PIL import Image
from custom import Identity


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO architecture

    def forward(self, x):
        pass


class attention_based_captioner(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # initializing feature-exctracion models
        # deleting the last FC layer (not sure about the dropout tho)
        # since we need the cnn features
        self.cnn_extractor = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.cnn_extractor.dropout = Identity()
        self.cnn_extractor.fc = Identity()
        # freezing inception
        for param in self.cnn_extractor.parameters():
            param.requires_grad = False

        self.detector = torch.hub.load(
            'ultralytics/yolov5', 'yolov5s', pretrained=True)
        # freezing yolo
        for param in self.detector.parameters():
            param.requires_grad = False

        self.embedding = nn.Linear(2048, 256)  # eh?

        self.decoder = None  # TODO add

    def forward(self, x):
        cnn_features = self.detector(x)

        detections = self.detector(x)  # TODO apply importance factor
        # for each image we need its width, height, confidence and class
        # (indexes 2-5 in .xywh)

        tmp = detections.xywh[0].permute(1, 0)  # TODO fix for batched input
        importances = tmp[2]*tmp[3]*tmp[4]
        print(importances)


image_path = 'smth'
img = Image.open(image_path)
convert_tensor = transforms.Compose(  # seems to be alright for both models
    [transforms.Resize((640, 640)),
     transforms.ToTensor()]
)
model = attention_based_captioner()
tens = convert_tensor(img)
c, h, w = tens.shape  # h and w should be multiples of 32 so need to reshape
model(tens.view(-1, c, h, w))
