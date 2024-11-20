def save_image(image, name):
    import numpy as np
    from PIL import Image
    image = image[0].cpu().numpy()
    image = (image + 1.0) * 127.5
    image = image.clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image.transpose(1, 2, 0))
    image.save(name)
