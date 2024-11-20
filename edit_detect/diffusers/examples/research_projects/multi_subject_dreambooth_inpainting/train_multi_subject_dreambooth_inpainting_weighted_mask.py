def weighted_mask(masks):
    mask_arrays = [(np.array(mask) / 255) for mask in masks]
    weights = [random.random() for _ in masks]
    weights = [(weight / sum(weights)) for weight in weights]
    weighted_masks = [(mask * weight) for mask, weight in zip(mask_arrays,
        weights)]
    summed_mask = np.sum(weighted_masks, axis=0)
    threshold = 0.5
    result_mask = summed_mask >= threshold
    return Image.fromarray(result_mask.astype(np.uint8) * 255)
