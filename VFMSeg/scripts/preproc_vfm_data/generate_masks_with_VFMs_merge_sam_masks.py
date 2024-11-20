def merge_sam_masks(image_masks):
    num_of_masks = len(image_masks)
    merged_masks = []
    if num_of_masks > 0:
        merged_masks = np.zeros((image_masks[0].shape[0], image_masks[0].
            shape[1]), dtype=np.int16)
        for i in range(num_of_masks):
            merged_masks[image_masks[i]] = i + 1
        return merged_masks
    else:
        return []
