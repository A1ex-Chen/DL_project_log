def process_conv_multimage(self, raw_conv, image):
    if image is None:
        return raw_conv, image
    if not isinstance(image, (list, tuple)):
        return raw_conv, image
    image_seqs = []
    for conv in raw_conv:
        image_seqs.extend(conv['image_seq'] if 'image_seq' in conv else [])
    images = []
    for idx in image_seqs:
        images.append(image[idx])
    return raw_conv, images
