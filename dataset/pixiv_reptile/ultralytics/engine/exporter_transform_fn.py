def transform_fn(data_item) ->np.ndarray:
    """Quantization transform function."""
    data_item: torch.Tensor = data_item['img'] if isinstance(data_item, dict
        ) else data_item
    assert data_item.dtype == torch.uint8, 'Input image must be uint8 for the quantization preprocessing'
    im = data_item.numpy().astype(np.float32) / 255.0
    return np.expand_dims(im, 0) if im.ndim == 3 else im
