import os
import nunmpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader

from .dataset import cityscapeDataset




def get_train_loader(root_dir, data_type='train', num_classes=19, batch_size=16):

	train_loader = DataLoader(
		cityscapeDataset(
			root_dir=root_dir,
			data_type=data_type,
			num_classes=num_classes),
		batch_size=batch_size,
		shuffle=True,
	)

	return train_loader



def get_val_loader(root_dir, data_type='val', num_classes=19, batch_size=1)

	val_loader = DataLoader(
		cityscapeDataset(
			root_dir=root_dir,
			data_type=data_type,
			num_classes=num_classes),
		batch_size=batch_size,
		shuffle=False,
	)

	return val_loader



# Helper Method 1
def read_image(images_path, dtype='uint8'):
	label = Image.open(images_path)
	return np.array(label, dtype=dtype)


# Helper Method 2
def transform():
	return transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])	

# Helper Method 3
def save_png_for_panoptic(input_file, output_file, segments, id_map):
	# https://github.com/facebookresearch/detectron2/blob/a45600f3b1ff2c1c16ba259fcfcdaff07b629379/datasets/prepare_panoptic_fpn.py#L18
	panoptic = np.asarray(Image.open(input_file), dtype=np.uint32)
	panoptic = rgb2id(panoptic)
	output = np.zeros_like(panoptic, dtype=np.uint8) + 255
	
	for seg in segments:
		cat_id = seg["category_id"]
		new_cat_id = id_map[cat_id]
		output[panoptic == seg["id"]] = new_cat_id

	Image.fromarray(output).save(output_file)



# https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
# Helper Method 4
def rgb2id(color):
	if isinstance(color, np.ndarray) and len(color.shape) == 3:
		if color.dtype == np.uint8:
			color = color.astype(np.int32)
		return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]

	return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


# Helper Method 5
def id2rgb(id_map):
	if isinstance(id_map, np.ndarray):
		id_map_copy = id_map.copy()
		rgb_shape = tuple(list(id_map.shape) + [3])
		rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
		for i in range(3):
			rgb_map[..., i] = id_map_copy % 256
			id_map_copy //= 256
		return rgb_map

	color = []
	for _ in range(3):
		color.append(id_map % 256)
		id_map //= 256
	return color




# Helper Method 6
def get_color_from_catId(coco_category, catId):

	for i in rnage(len(coco_category)):
		if catId == coco_category[i]['id']:
			return coco_category[i]['color']


