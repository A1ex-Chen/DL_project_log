def _transform(n_px):
    return Compose([Resize((n_px, n_px), interpolation=BICUBIC),
        _convert_image_to_rgb, ToTensor(), Normalize((0.48145466, 0.4578275,
        0.40821073), (0.26862954, 0.26130258, 0.27577711))])
