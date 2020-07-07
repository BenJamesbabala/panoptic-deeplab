import torch
import torch.nn as nn




class semDecoder(object):
	"""
	docstring for semDecoder
	"""
	
	def __init__(self):
		super(semDecoder, self, in_channels=256*5).__init__()

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