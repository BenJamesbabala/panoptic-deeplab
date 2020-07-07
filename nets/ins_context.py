import torch
import torch.nn as nn




class insContextBlock(nn.Module):
	"""
	docstring for insContextBlock
	"""
	
	def __init__(self):
		super(insContextBlock, self, in_channels=2048).__init__()
	
		self.conv1x1 = nn.Conv2d(
			in_channels = in_channels,
			out_channels = 256,
			kernel_size = 1,
			stride = 1,
			padding = 0,
			dilation = 1,
			bias=False
		)
		self.bn1x1 = nn.BatchNorm2d(
			num_features = 256,
			momentum = 1e-3
		)

		self.conv3x3_6 = nn.Conv2d(
			in_channels = in_channels,
			out_channels = 256,
			kernel_size = 3,
			stride = 1,
			padding = 6,
			dilation = 6,
			bias=False
		)
		self.bn3x3_6 = nn.BatchNorm2d(
			num_features = 256,
			momentum = 1e-3
		)

		self.conv3x3_12 = nn.Conv2d(
			in_channels = in_channels,
			out_channels = 256,
			kernel_size = 3,
			stride = 1,
			padding = 12,
			dilation = 12,
			bias=False
		)
		self.bn3x3_12 = nn.BatchNorm2d(
			num_features = 256,
			momentum = 1e-3
		)

		self.conv3x3_18 = nn.Conv2d(
			in_channels = in_channels,
			out_channels = 256,
			kernel_size = 3,
			stride = 1,
			padding = 18,
			dilation = 18,
			bias=False
		)
		self.bn3x3_18 = nn.BatchNorm2d(
			num_features = 256,
			momentum = 1e-3
		)

		self.conv1x1_iPool = nn.Conv2d(
			in_channels = in_channels,
			out_channels = 256,
			kernel_size = 1,
			stride = 1,
			padding = 0,
			dilation = 1,
			bias = False
		)
		self.bn1x1_iPool = nn.BatchNorm2d(
			num_features = 256,
			momentum = 1e-3
		)

		self.relu = nn.ReLU(inplace=true)

		def forward(self, x):

			x1 = self.conv1x1(x)
			x1 = self.bn1x1(x1)
			x1 = self.relu(x1)

			x2 = self.conv3x3_6(x)
			x2 = self.bn3x3_6(x2)
			x2 = self.relu(x2)

			x3 = self.conv3x3_12(x)
			x3 = self.bn3x3_12(x3)
			x3 = self.relu(x3)

			x4 = self.conv3x3_18(x)
			x4 = self.bn3x3_18(x4)
			x4 = self.relu(x4)

			# This is the image pooling (gloabl avg pooling + conv layers)
			x5 = nn.avg_pool2d(x, x.size()[2:]) 
			x5 = self.conv1x1_iPool(x5)
			x5 = self.bn1x1_iPool(x5)
			x5 = nn.Upsample(x.shape[2], x.shape[3], mode='bilinear', align_corners=True)(x5)

			x = torch.cat((x1, x2, x3, x4, x5), 1) # Concatenate all different features

			return x
