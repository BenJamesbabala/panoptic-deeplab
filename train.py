import os
import numpy as np
import glob
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from nets import ResnetBackbone, PSDL
from utils import get_train_loader, get_val_loader, compute_loss

###################################
LOG_INTERVAL = 10


###################################
# Getting the deivce here		  #	
dev='cpu' # Default setting		  #
if torch.cuda.is_available():	  #
	dev='cuda'					  #	
###################################


def train_epoch(model, epoch, train_data_loader, optimizer):
	
	model.train()
	for param in model.parameters():  # Setting complete model to be trainable
		param.requires_grad = True

	learning_rate = 0.0
	if epoch < 30:
		for param_group in optimizer.param_groups:
			param_group['lr'] = 1e-2
			learning_rate = param_group['lr']
	if epoch >= 30 and epoch < 60:
		for param_group in optimizer_backbone.param_groups:
			param_group['lr'] = 1e-3
			learning_rate = param_group['lr']
	if epoch >= 60:
		for param_group in optimizer_backbone.param_groups:
			param_group['lr'] = 1e-4
			learning_rate = param_group['lr']

	print("epoch number --> " + str(epoch) + " learning rate ---> " + str(learning_rate))

	train_loss = 0
	for batch_idx, (data, target) in enumerate(train_data_loader):
		
		data = data.to(dev)
		target = target.to(dev)

		optimizer.zero_grad()

		sempred, inspred, insreg = model(data)
		
		loss = compute_loss(sempred, inspred, insreg, target) 

		loss.backward()

		train_loss += loss.item()

		optimizer.step()

	print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))


def val_epoch(model, epoch, val_data_loader):
	# Need to implement...

	pass



def get_args():
	parser = argparse.ArgumentParser(description='Panoptic DeepLab training script.')
	parser.add_argument('-r', '--root_dir', type=str, required=True, help='Path to the CityScape folder')
	parser.add_argument('-e', '--num_epcohs', type=int, default=100, help='Number of epochs.')

	args = parser.parse_args()
	return args



if __name__ == '__main__':

	args = get_args()

	# Loading the complete architecture here
	model = PSDL().to(dev)
	print(model)

	train_data_loader = get_train_loader(args.root_dir)
	val_data_loader = get_val_loader(args.root_dir)

	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

	# Training...
	for epoch in range(args.num_epcohs):
		train_epoch(model, epoch, train_data_loader, optimizer)
		val_epochO(model, epoch, val_data_loader)

	torch.save(model.state_dict(), "models/epoch" + str(args.num_epcohs) + ".pth")

	print("Training Done!!!")













