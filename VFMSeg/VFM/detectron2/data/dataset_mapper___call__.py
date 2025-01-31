def __call__(self, dataset_dict):
    """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format=self.
        image_format)
    utils.check_image_size(dataset_dict, image)
    if 'sem_seg_file_name' in dataset_dict:
        sem_seg_gt = utils.read_image(dataset_dict.pop('sem_seg_file_name'),
            'L').squeeze(2)
    else:
        sem_seg_gt = None
    aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
    transforms = self.augmentations(aug_input)
    image, sem_seg_gt = aug_input.image, aug_input.sem_seg
    image_shape = image.shape[:2]
    dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.
        transpose(2, 0, 1)))
    if sem_seg_gt is not None:
        dataset_dict['sem_seg'] = torch.as_tensor(sem_seg_gt.astype('long'))
    if self.proposal_topk is not None:
        utils.transform_proposals(dataset_dict, image_shape, transforms,
            proposal_topk=self.proposal_topk)
    if not self.is_train:
        dataset_dict.pop('annotations', None)
        dataset_dict.pop('sem_seg_file_name', None)
        return dataset_dict
    if 'annotations' in dataset_dict:
        self._transform_annotations(dataset_dict, transforms, image_shape)
    return dataset_dict
