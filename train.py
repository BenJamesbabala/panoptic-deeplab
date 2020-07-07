from nets import ResnetBackbone

import torch
import torch.nn as nn


# Getting the deivce here
dev='cpu' # Default setting
if torch.cuda.is_available():
	dev='cuda'


# Loading the complete architecture here
model = ResnetBackbone().to(dev)
print(model)


