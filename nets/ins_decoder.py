import torch
import torch.nn as nn




class insDecoder(nn.Module):
	"""
	docstring for insDecoder
	"""
	
	def __init__(self, in_channels_context=256*5, in_channels_block1=256, in_channels_block2=512):
		super(insDecoder, self).__init__()

		self.conv1x1_1 = nn.Conv2d( # For the features coming from the corresponding Context block
			in_channels = in_channels_context,
			out_channels = 256,
			kernel_size = 1,
			stride = 1,
			padding = 0,
			dilation = 1,
			bias=False
		)
		self.bn1x1_1 = nn.BatchNorm2d(
			num_features = 256,
			momentum = 1e-3
		)

		self.conv1x1_2 = nn.Conv2d( # For the feature coming from the 1/4th size feature block from encoder
			in_channels = in_channels_block1,
			out_channels = 16,
			kernel_size = 1,
			stride = 1,
			padding = 0,
			dilation = 1,
			bias=False
		)
		self.bn1x1_2 = nn.BatchNorm2d(
			num_features = 16,
			momentum = 1e-3
		)

		self.conv1x1_3 = nn.Conv2d( # For the feature coming from the 1/8th size fetaure block from encoder
			in_channels = in_channels_block2,
			out_channels = 32,
			kernel_size = 1,
			stride = 1,
			padding = 0,
			dilation = 1,
			bias=False
		)
		self.bn1x1_3 = nn.BatchNorm2d(
			num_features = 32,
			momentum = 1e-3
		)

		self.dwconv5x5_1 =  nn.Conv2d(
			in_channels = 288, # 288 comes from the 256(from the context) + 32(from the 1/8th)
			out_channels = 288,
			kernel_size = 5,
			stride = 1,
			padding = 2, # To keep the same input size
			dilation = 1,
			groups = 288,
			bias = False
		)
		self.dwbn5x5_1 = nn.BatchNorm2d(
			num_features = 288,
			momentum = 1e-3
		)

		self.pwconv_1 = nn.Conv2d(
			in_channels = 288,
			out_channels = 128,
			kernel_size = 1,
			stride = 1,
			padding = 0,
			dilation = 1,
			groups = 1,
			bias = False
		)
		self.pwbn_1 = nn.BatchNorm2d(
			num_features = 128,
			momentum = 1e-3
		)


		self.dwconv5x5_2 =  nn.Conv2d(
			in_channels = 144, # 288 comes from the 128(from the context) + 16(from the 1/4th)
			out_channels = 144,
			kernel_size = 5,
			stride = 1,
			padding = 2,
			dilation = 1,
			groups = 144,
			bias = False
		)
		self.dwbn5x5_2 = nn.BatchNorm2d(
			num_features = 144,
			momentum = 1e-3
		)
		self.pwconv_2 = nn.Conv2d(
			in_channels = 144,
			out_channels = 128,
			kernel_size = 1,
			stride = 1,
			padding = 0,
			dilation = 1,
			groups = 1,
			bias = False
		)
		self.pwbn_2 = nn.BatchNorm2d(
			num_features = 128,
			momentum = 1e-3
		)

		self.relu = nn.ReLU(inplace=True)


		def forward(x, x1_8, x1_4):

			x = self.conv1x1_1(x)
			x = self.bn1x1_1(x)
			x = self.relu(x)

			# First Unsampling (to the block of 1/8th size)
			x = nn.Upsample(x1_8.shape[2], x1_8.shape[3], mode='bilinear', align_corners=True)(x)

			# Red Line Conv Layer
			x1_8 = self.conv1x1_3(x1_8)
			x1_8 = self.bn1x1_3(x1_8)
			x1_8 = self.relu(x1_8)

			# Concatenating
			x = torch.cat((x, x1_8), 1) # Concatenating the first unsample and x1/8th(red line) output

			# First DepthWise Separable Conv layer
			x = self.dwconv5x5_1(x)
			x = self.dwbn5x5_1(x)
			x = self.relu(x)
			x = self.pwconv_1(x)
			x = self.pwbn_1(x)
			x = self.relu(x)

			# Second Unsampling (to the block of 1/4th size)
			x = nn.Upsample(x1_4.shape[2], x1_4.shape[3], mode='bilinear', align_corners=True)(x)

			# Blue Line Conv Layer
			x1_4 = self.conv1x1_2(x1_4)
			x1_4 = self.bn1x1_2(x1_4)
			x1_4 = self.relu(x1_4)

			# Concatenating
			x = torch.cat((x, x1_4), 1) # Concatenating the second unsample and x1/4th(blue line) output

			# Second DepthWise Separable Conv layer
			x = self.dwconv5x5_2(x)
			x = self.dwbn5x5_2(x)
			x = self.relu(x)
			x = self.pwconv_2(x)
			x = self.pwbn_2(x)
			x = self.relu(x)
			

			return x



