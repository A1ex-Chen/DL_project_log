def convert_batched_inputs_to_c2_format(batched_inputs, size_divisibility,
    device):
    """
    See get_caffe2_inputs() below.
    """
    assert all(isinstance(x, dict) for x in batched_inputs)
    assert all(x['image'].dim() == 3 for x in batched_inputs)
    images = [x['image'] for x in batched_inputs]
    images = ImageList.from_tensors(images, size_divisibility)
    im_info = []
    for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
        target_height = input_per_image.get('height', image_size[0])
        target_width = input_per_image.get('width', image_size[1])
        scale = target_height / image_size[0]
        im_info.append([image_size[0], image_size[1], scale])
    im_info = torch.Tensor(im_info)
    return images.tensor.to(device), im_info.to(device)
