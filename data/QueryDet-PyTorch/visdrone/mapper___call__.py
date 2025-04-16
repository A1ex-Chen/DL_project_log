def __call__(self, dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format=self.img_format)
    utils.check_image_size(dataset_dict, image)
    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
    image_shape = image.shape[:2]
    dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.
        transpose(2, 0, 1)))
    if not self.is_train:
        dataset_dict.pop('annotations', None)
        return dataset_dict
    if 'annotations' in dataset_dict:
        for anno in dataset_dict['annotations']:
            anno.pop('segmentation', None)
            anno.pop('keypoints', None)
        annos = [utils.transform_instance_annotations(obj, transforms,
            image_shape) for obj in dataset_dict.pop('annotations') if obj.
            get('iscrowd', 0) == 0]
        instances = utils.annotations_to_instances(annos, image_shape,
            mask_format=self.mask_format)
        dataset_dict['instances'] = utils.filter_empty_instances(instances)
    return dataset_dict
