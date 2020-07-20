import torch
import torch.nn as nn

class insRegression(nn.Module):
	"""
	docstring for insRegression
	"""

	def __init__(self):
		super(insRegression, self, num_classes=2).__init__()
		self.num_classes = num_classes

		self.conv5x5 = nn.Conv2d(
			in_channels = 128,
			out_channels = 32,
			kernel_size = 5,
			stride = 1,
			padding = 2,
			dilation = 1,
			bias = False
		)
		self.bn5x5 = nn.BatchNorm2d(
			num_features = 32,
			momentum = 1e-3
		)

		self.conv1x1 = nn.Conv2d(
			in_channels = 32,
			out_channels = self.num_classes,
			kernel_size = 1,
			stride = 1,
			padding = 0,
			dilation = 1,
			bias = False
		) 
		self.bn1x1 = nn.BatchNorm2d(
			num_features = self.num_classes,
			momentum = 1e-3
		)

		self.relu = nn.ReLU(inplace=True)
		

		def forward(x):
			x = self.conv5x5(x)
			x = self.bn5x5(x)
			x = self.relu(x)

			x = self.conv1x1(x)
			x = self.bn1x1(x)
			x = self.relu(x)

			return x



