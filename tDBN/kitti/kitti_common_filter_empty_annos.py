def filter_empty_annos(image_annos):
    new_image_annos = []
    for anno in image_annos:
        if anno['name'].shape[0] > 0:
            new_image_annos.append(anno.copy())
    return new_image_annos
