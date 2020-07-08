import os

import torch
import torchvision.datasets as datasets
import torch.nn.functional as F


def load_cityscape():

	return datasets.Cityscapes('./data/cityscapes', split='train', mode='fine', target_type=['semantic','instance'])



class datasetCityScape(object):
	"""
	docstring for datasetCityScape
	"""
	def __init__(self):
		super(datasetCityScape, self).__init__()



		self.data = load_cityscape()		