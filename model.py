import numpy as np
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from yolov4.tf import YOLOv4

from custom import Identity


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO architecture

    def forward(self, x):
        pass


class attention_based_captioner(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # probably gonna use InceptionV3 as classifier
        # deleting the last FC layer (not sure about the dropout tho) since we need the cnn features
        self.cnn_extractor = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.cnn_extractor.dropout = Identity()
        self.cnn_extractor.fc = Identity()
        for param in self.cnn_extractor.parameters():
            param.requires_grad = False

        self.detector = YOLOv4()

        self.embedding = nn.Linear(2048, 256)  # eh?

        self.decoder = None  # TODO add

    def forward(self, x):
        cnn_features = self.detector(x)
        # prolly will need to change it depending on the shape (to apply IF and make concat possible)
        detector_features = self.detector(x)  # TODO apply importance factor

        image_info = torch.concat(cnn_features, detector_features)
        image_info = self.embedding(image_info)
        # TODO attention factor
        return self.decoder(image_info)  # ?
