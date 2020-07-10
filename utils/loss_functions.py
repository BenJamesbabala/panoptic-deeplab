import torch
import torch.nn as nn



def weighted_bootstrapped_CEL(predictions, labels, weight=1.0, K=0.15, N=641*641):
	'''
	The Deafult args are for the 'COCO' dataset from the paper. 
	They train it on the Images of size (641*641)
	'''
 

	weighted_CELoss = weight * nn.CrossEntropyLoss((predictions, labels))
	weighted_CELoss = weighted_CELoss.view(-1)

	total_elements_in_CELoss = weighted_CELoss.numel() # Getting number of elements
	print("number of pixels --> ", N, " from numel --> ", total_elements_in_CELoss)

	K_number_pixels = int(K * total_elements_in_CELoss)
	top_K_loss_values = torch.topk(weighted_CELoss, K_number_pixels, largest=True)
	
	return top_K_loss_values.mean()



def mse_loss(predicted_center, gt_center):
	'''
	A very simple MSE Loss between the instances centers (ground_truth vs predicted)
	'''
	
	return nn.MSELoss(predicted_center, gt_center)



def l1_loss(pixels_coords, center_coords):
	'''
	A very simple L1 Loss between the pixels offsets from the instance centers
	'''
	
	return nn.L1Loss(pixels_coords, center_coords)
	
















