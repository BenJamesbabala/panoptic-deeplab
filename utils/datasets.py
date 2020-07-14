import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F



class COCOPanopticDataset(Dataset):
	"""
	docstring for COCOPanopticDataset
	root_dir: Is the root_dir (path to the coco folder)
	"""

	def __init__(self, root_dir, transform, data_type='train'):
		super(COCOPanopticDataset, self).__init__()

		self.root_dir = root_dir # Path to the COCO folder
		self.data_type = data_type # Train or Val

		self.images_list = [] # I will add numpy array of the images with .jpg or .png extentsion
		self.labels_list = [] # I will add labels of pixels for each image in the images_list

		self.transform = transform # Transformation to do on the dataset

		



	def __getitem__(self, index):
		if index > self.__len__():
			raise StopIteration
		return images_list[index], labels_list[index]


	def __len__(self):
		return len(self.labels_list)
				

	















