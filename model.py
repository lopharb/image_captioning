import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.transforms = {
            'to_tensor_1280': transforms.Compose(  # seems to be alright for both models
                [transforms.Resize((1280, 1280)),
                 transforms.ToTensor()]
            ),
            'aumgents':
            transforms.Compose(
                [None]  # TODO think what i wanted to add there :D
            )
        }
        # initializing feature-exctracion models
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

        self.decoder = Decoder()

    def forward(self, x: Image):
        """
        this expects PIL Image as input
        it then undergoes different conversions to match the networks requirements 
        """

        detections = self.detector(x)
        tmp = detections.xywh[0].permute(1, 0)
        importances = tmp[2]*tmp[3]*tmp[4]
        concatted = torch.concat(
            [detections.xywh[0], importances.view(importances.shape[0], -1)],
            dim=1
        ).flatten()
        # concatted is: x_center, y_center, width, height, confidence, class_id, importance factor

        x_tens = self.transforms['to_tensor_1280'](x)
        c, h, w = x_tens.shape  # h and w should be multiples of 32 so need to reshape
        cnn_features = self.cnn_extractor(x_tens.view(-1, c, h, w))
        # TODO need to work around the shape bc it doesn't seem to be right
        print(cnn_features[0].shape)
        # (indexes 2-5 in .xywh)
        return F.pad(concatted, pad=(0, 2048-len(concatted))).view(1, 2048)


if __name__ == '__main__':
    model = attention_based_captioner()
    transf = transforms.Compose(  # seems to be alright for both models
        [transforms.Resize((640, 640))]
    )
    img = 'data/bicycle.jpg'
    img = transf(Image.open(img))
    res = model(img)
    print(res.shape)
