from nets import ResnetBackbone, PSDL
from utils import datasetCityScape

import torch
import torch.nn as nn


# Getting the deivce here
dev='cpu' # Default setting
if torch.cuda.is_available():
	dev='cuda'

# Loading the complete architecture here
# model = ResnetBackbone().to(dev)
model = PSDL().to(dev)
print(model)


