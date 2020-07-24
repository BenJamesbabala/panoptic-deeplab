import torch
import torch.nn as nn



def weighted_bootstrapped_CEL(predictions, labels, K=0.15, N=1025*2049):
	'''
	The Deafult args are for the 'COCO' dataset from the paper. 
	They train it on the Images of size (641*641)
	'''
 
	criterion = nn.CrossEntropyLoss()
	weighted_CELoss = criterion(predictions, labels)
	weighted_CELoss = weighted_CELoss.view(-1)

	total_elements_in_CELoss = weighted_CELoss.numel() # Getting number of elements
	print("number of pixels --> ", N, " from numel --> ", total_elements_in_CELoss)

	K_number_pixels = int(K * total_elements_in_CELoss)

	return top_K_loss_values



def mse_loss(predicted_center, gt_center):
	'''
	A very simple MSE Loss between the instances centers (ground_truth vs predicted)
	'''
	
	criterion = nn.MSELoss()
	return criterion(predicted_center, gt_center)



def l1_loss(pixels_coords, center_coords):
	'''
	A very simple L1 Loss between the pixels offsets from the instance centers
	'''
	criterion = nn.L1Loss()
	return criterion(pixels_coords, center_coords)
	

def upsample_preds(sempred, inspred, insreg, upsample):
	'''
		We need to upsample the predictions to get similar size as input images
	'''

	return upsample(sempred), upsample(inspred), upsample(insreg)


def compute_loss(sempred, inspred, insreg, target, weight_semp=1.0, weight_insp=1.0, weight_insr=1.0):

	'''
	Compute the total loss value from all three components of the architecture
	'''
	input_shape = [target['semantic_img'].numpy().shape[0], target['semantic_img'].numpy().shape[1]]

	upsample = nn.Upsample((input_shape[0], input_shape[1]), mode='bilinear', align_corners=True)

	sempred, inspred, insreg = upsample_preds(sempred, inspred, insreg, upsample)

	semp_loss = weighted_bootstrapped_CEL(sempred, target['semantic_img']) * target['sem_weights']

	insp_loss = mse_loss(inspred, target['centers']) * target['center_weights']

	insr_loss = l1_loss(insreg, target['offsets']) * target['offset_weights']

	total_loss = weight_semp*semp_loss + weight_insp*insp_loss + weight_insr*insr_loss

	return total_loss














