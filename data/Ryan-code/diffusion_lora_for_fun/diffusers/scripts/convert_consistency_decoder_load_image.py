def load_image(uri, size=None, center_crop=False):
    import numpy as np
    from PIL import Image
    image = Image.open(uri)
    if center_crop:
        image = image.crop(((image.width - min(image.width, image.height)) //
            2, (image.height - min(image.width, image.height)) // 2, (image
            .width + min(image.width, image.height)) // 2, (image.height +
            min(image.width, image.height)) // 2))
    if size is not None:
        image = image.resize(size)
    image = torch.tensor(np.array(image).transpose(2, 0, 1)).unsqueeze(0
        ).float()
    image = image / 127.5 - 1.0
    return image
