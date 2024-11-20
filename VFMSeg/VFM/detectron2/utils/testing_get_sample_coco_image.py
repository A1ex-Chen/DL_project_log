def get_sample_coco_image(tensor=True):
    """
    Args:
        tensor (bool): if True, returns 3xHxW tensor.
            else, returns a HxWx3 numpy array.

    Returns:
        an image, in BGR color.
    """
    try:
        file_name = DatasetCatalog.get('coco_2017_val_100')[0]['file_name']
        if not PathManager.exists(file_name):
            raise FileNotFoundError()
    except IOError:
        file_name = PathManager.get_local_path(
            'http://images.cocodataset.org/train2017/000000000009.jpg')
    ret = read_image(file_name, format='BGR')
    if tensor:
        ret = torch.from_numpy(np.ascontiguousarray(ret.transpose(2, 0, 1)))
    return ret
