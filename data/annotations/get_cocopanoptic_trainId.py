import json
import copy


if __name__ == '__main__':
	with open('panoptic_val2017.json', "r") as f:
		coco_json = json.load(f)


	coco_trainid_json = copy.deepcopy(coco_json)
	coco_cats = coco_json.pop('categories')
	coco_anns = coco_json.pop('annotations')
	print(coco_cats)

	coco_train_id_to_eval_id = [coco_cat['id'] for coco_cat in coco_cats]

	coco_eval_id_to_train_id = {v: k for k, v in enumerate(coco_train_id_to_eval_id)}

	new_cats = []
	for coco_cat in coco_cats:
		coco_cat['id'] = coco_eval_id_to_train_id[coco_cat['id']]
		new_cats.append(coco_cat)

	coco_trainid_json['categories'] = new_cats


	new_anns = []

	for coco_ann in coco_anns:
		segments_info = coco_ann.pop('segments_info')
		new_segments_info = []

		for segment_info in segments_info:
			segment_info['category_id'] = coco_eval_id_to_train_id[segment_info['category_id']]
			new_segments_info.append(segment_info)

		coco_ann['segments_info'] = new_segments_info
		new_anns.append(coco_ann)

	coco_trainid_json['annotations'] = new_anns

	with open('panoptic_train2017_trainId.json', 'w') as f:
		json.dump(coco_trainid_json, f)



