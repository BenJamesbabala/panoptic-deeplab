import numpy as np

import torch
import torch.nn as nn



def weighted_bootstrapped_CEL(predictions, labels, weight=1.0, K=0.15, N=1025*2049):
	'''
	The Deafult args are for the 'Cityscape' dataset from the paper. 
	They train it on the Images of size (1025*2049)
	'''
 

	weighted_CELoss = weight * nn.CrossEntropyLoss((predictions, labels))
	weighted_CELoss = weighted_CELoss.view(-1)

	total_elements_in_CELoss = weighted_CELoss.numel() # Getting number of elements
	print("number of pixels --> ", N, " from numel --> ", total_elements_in_CELoss)
	
	K_number_pixels = int(K * total_elements_in_CELoss)
	top_K_loss_values = torch.topk(weighted_CELoss, K_number_pixels, largest=True)
	
	return top_K_loss_values.mean()


def function():
	pass


