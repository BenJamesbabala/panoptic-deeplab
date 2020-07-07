import torch.nn as nn
from torchvision import models


class ResnetBackbone(nn.Module):
	"""
	Extracting the features using the Resnet50 as the Backbone

		1. Removing the last output layer and taking the rest to extract features

		self.conv1: 224 --> 112  {1/2}
		self.block1: 112 --> 56  {1/4}
		self.block2: 56 --> 28	 {1/8}
		self.block3: 28 --> 14 	 {1/16}
		self.block4: 14 --> 14   {1/16}
	"""

	def __init__(self):
		super(ResnetBackbone, self).__init__()

		resnet = models.resnet50(pretrained=True)

		self.conv1 = nn.Sequential(
			resnet.conv1,
			resnet.bn1,
			resnet.relu,
			resnet.maxpool
		)

		self.block1 = nn.Sequential(
			*list(resnet.layer1)
		)
		self.block2 = nn.Sequential(
			*list(resnet.layer2)
		)
		self.block3 = nn.Sequential(
			*list(resnet.layer3)
		)
		self.block4 = nn.Sequential(
			*list(resnet.layer4)
		)
		# We are changing the stride for the last block to keep the stride ratio(with original size) to 1/16(same as block 3)
		self.block4[0].downsample = nn.Sequential(
										nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
										nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) 


	def forward(self, x):
		x0 = self.conv1(x)  
		x1 = self.block1(x0) # Blue Line goes to the decoders
		x2 = self.block2(x1) # Red Line goes to the decoders
		x3 = self.block3(x2)
		x4 = self.block4(x3) # Output of encoder goes to the context blocks

		return x1, x2, x4

