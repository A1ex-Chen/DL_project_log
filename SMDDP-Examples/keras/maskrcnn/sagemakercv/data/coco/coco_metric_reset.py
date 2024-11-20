def reset(self, dataset):
    """Reset the dataset and groundtruth data index in this object.

    Args:
      dataset: dict of groundtruth data. It should has similar structure as the
        COCO groundtruth JSON file. Must contains three keys: {'images',
          'annotations', 'categories'}.
        'images': list of image information dictionary. Required keys: 'id',
          'width' and 'height'.
        'annotations': list of dict. Bounding boxes and segmentations related
          information. Required keys: {'id', 'image_id', 'category_id', 'bbox',
            'iscrowd', 'area', 'segmentation'}.
        'categories': list of dict of the category information.
          Required key: 'id'.
        Refer to http://cocodataset.org/#format-data for more details.

    Raises:
      AttributeError: If the dataset is empty or not a dict.
    """
    assert dataset, 'Groundtruth should not be empty.'
    assert isinstance(dataset, dict
        ), 'annotation file format {} not supported'.format(type(dataset))
    self.anns, self.cats, self.imgs = dict(), dict(), dict()
    self.dataset = copy.deepcopy(dataset)
    self.createIndex(use_ext=True)
