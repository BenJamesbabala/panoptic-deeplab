import torch
import torch.nn as nn

class postProcessing(nn.Module):
	"""
	docstring for postProcessing
	"""
	def __init__(self, arg):
		super(postProcessing, self).__init__()

		self.arg = arg
		

	def forward(self, x):

		return x