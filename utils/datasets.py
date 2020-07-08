import os

import torch
import torchvision.datasets as datasets
import torch.nn.functional as F


def load_cityscape():

	datasets.Cityscapes('./data/cityscapes', split='train', mode='fine', target_type=['semantic','instance'])

	return


class datasetCityScpape(object):
	"""
	docstring for datasetCityScpape
	"""
	def __init__(self):
		super(datasetCityScpape, self).__init__()



		self.data = load_cityscape()		