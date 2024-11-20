@staticmethod
def _segment_image(image, bbox):
    """Segments the given image according to the provided bounding box coordinates."""
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    x1, y1, x2, y2 = bbox
    segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new('RGB', image.size, (255, 255, 255))
    transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1
        ]), dtype=np.uint8)
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image
