import numpy as np
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

from custom import Identity
# probably gonna use InceptionV3 as classifier


# deleting the last FC layer (not sure about the dropout tho) since we need the cnn features
classifier = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
classifier.dropout = Identity()
classifier.fc = Identity()
x = torch.rand(1, 3, 299, 299)
classifier.eval()
print(classifier(x).shape)

# TODO add the detector and decoder
