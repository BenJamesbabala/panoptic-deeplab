import os
import numpy as np
import glob
import json
from PIL import Image

import torch
from torch.utils.data import Dataset


CITYSCAPE_TRAIN_TO_EVAL = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
CITYSCAPES_THINGS_CATID = [11, 12, 13, 14, 15, 16, 17, 18]


class cityscapeDataset(Dataset):
	"""
	docstring for cityscapeDataset
	"""

	def __init__(self, root_dir, data_type, rgb2id, transform, num_classes=19):
		super(cityscapeDataset, self).__init__()

		self.root_dir = root_dir # Path to the COCO folder
		self.data_type = data_type # Train or Val or Test
        self.num_classes = num_classes


		self.images_list = []
		self.anns_list = []
		# self.data = [] 
		# self.labels = []


        # self.images_list = sorted(glob.glob(os.path.join(self.root_dir, 'leftImg8bit', data_type, '*_leftImg8bit.png')))
        # self.anns_list = sorted(glob.glob(os.path.join(self.root_dir, 'gtFine', data_type, '*_gtFine_labelTrainIds.png')))

        json_filename = os.path.join(self.root_dir, 'gtFine', 'cityscapes_panoptic_' + self.data_type + '_trainId.json')
        panoptic_info = json.load(open(json_filename))

		for img in panoptic_info['images']:
			img_name = img['file_name']
			img_name = os.path.join(self.root_dir, 'leftImg8bit', self.data_type, img_name.split('_')[0], img_name.replace('_gtFine', ''))
			self.images_list.append(img_name)

		for ann in panoptic_info['annotations']:
			ann_name = ann['file_name']
			ann_name = os.path.join(self.root_dir, 'gtFine', 'cityscapes_panoptic_' + self.data_type + '_trainId',  ann_name)
			self.anns_list.append([ann_name, ann['segments_info']])

        self.images_list = sorted(self.images_list, key=lambda x: x[0]) 
        self.anns_list = sorted(self.anns_list, key=lambda x: x[0])


    def __len__(self):
        return len(self.images_list)


	def __getitem__(self, index):
		if index > self.__len__():
			raise StopIteration


        img = Image.open(self.images_list[index]).convert('RGB')
        img_arr  = np.asarray(img) 

        label = Image.open(self.anns_list[index][0])
        label_arr = np.asarray(label, dtype=np.uint8)


        # Need to check if the STD and MEAN will change labels and IDs
        # if self.data_type == 'train':
        #     img_arr = transform(img_arr)
        #     label_arr = transform(label_arr)


        # Getting the panoptic labels here...
        panoptic_converted_labels = self.get_panoptic_labels(label_arr, self.anns_list[index][1])


        return img_arr, panoptic_converted_labels


    def get_panoptic_labels(label_arr, segments_info):

        label_id_img = rgb2id(label_arr)
        semantic_img = np.zeros_like(label_id_img, dtype=np.uint8) #+ 255
        instance_img = np.zeros_like(label_id_img, dtype=np.uint8)

        center_each_pixel = np.zeros((label_id_img.shape[0], label_id_img.shape[1], 1), dtype=np.float32)
        offset_each_pixel = np.zeros((label_id_img.shape[0], label_id_img.shape[1], 2), dtype=np.float32)

        sem_weight_each_pixel = np.ones((label_id_img.shape[0], label_id_img.shape[1], 1), dtype=np.uint8)
        center_weight_each_pixel = np.zeros((label_id_img.shape[0], label_id_img.shape[1], 1), dtype=np.uint8)
        offset_weight_each_pixel = np.zeors((label_id_img.shape[0], label_id_img.shape[1], 1), dtype=np.uint8)


        std = 8 # 8 Pixels from the paper
        x = np.arange(0, 6*std, 1, dtype=np.float32).reshape(1, 6*std)
        y = np.arange(0, 6*std, 1, dtype=np.float32).reshape(6 * std, 1)
        x0, y0 = 3*std, 3*std
        gauss_dist = np.exp((- ((x-x0)**2 + (y-y0)**2))/ (2*std**2))

        for seg in segments_info:
            if seg['iscrowd'] == 0:
                semantic_img[label_id_img == seg['id']] = seg['category_id']
                center_weight_each_pixel[label_id_img == seg["id"]] = 1
                offset_weight_each_pixel[label_id_img == seg["id"]] = 1


            if seg['category_id'] in CITYSCAPES_THINGS_CATID:
                instance_img[label_id_img == seg['id']] = 1

                ins_pixels = np.where(label_id_img == seg["id"])
                center_y = int(np.mean(ins_pixels[0]))
                center_x = int(np.mean(ins_pixels[1]))

                ins_area = len(ins_pixels)
                if ins_area < 64*64: # From Paper
                    sem_weight_each_pixel[label_id_img == seg('id')] = 3

                # Add the 2D Gaussian to the center and handle the corners

                xmin = int(center_x - 3 * 8)
                ymin = int(center_y - 3 * 8)
                xmax = int(center_x + 3 * 8)
                ymax = int(center_y + 3 * 8)

                # Outside image pixel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(label_id_img.shape[1], xmax)
                ymax = min(label_id_img.shape[0], ymax)

                # Generating Gaussian Dist. around the center

                xs = max(0, -xmin)
                xe = min(xmax, label_id_img.shape[1]) - xmin
                ys = max(0, -ymin)
                ye = min(ymax, label_id_img.shape[0]) - ymin

                center_each_pixel[ymin:ymax, xmin:xmax, 0]  = np.maximum(gauss_dist[ys:ye, xs:xe], center_each_pixel[ymin:ymax, xmin:xmax, 0])


                # Getting Offset from the centers

                y_idxs = np.cumsum((np.ones((label_id_img.shape[0], label_id_img.shape[1]), dtype=np.float32)), axis=0) - 1
                x_idxs = np.cumsum((np.ones((label_id_img.shape[0], label_id_img.shape[1]), dtype=np.float32)), axis=1) - 1

                offset_each_pixel[ins_pixels[0], ins_pixels[1], np.zeros_like(ins_pixels[0])] = center_x - x_idxs[ins_pixels]
                offset_each_pixel[ins_pixels[0], ins_pixels[1], np.ones_like(ins_pixels[1])] = center_y - y_idxs[ins_pixels]


        labels = {
            'semantic_img': torch.from_numpy(semantic_img),
            'instance_img': torch.from_numpy(instance_img),
            'centers': torch.from_numpy(center_each_pixel),
            'offsets': torch.from_numpy(offset_each_pixel),
            'sem_weights': torch.from_numpy(sem_weight_each_pixel),
            'center_weights': torch.from_numpy(center_weight_each_pixel),
            'offset_weights': torch.from_numpy(offset_weight_each_pixel)
        }

        return labels




