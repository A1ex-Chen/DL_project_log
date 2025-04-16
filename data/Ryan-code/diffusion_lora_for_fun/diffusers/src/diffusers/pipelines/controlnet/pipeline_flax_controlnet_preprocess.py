def preprocess(image, dtype):
    image = image.convert('RGB')
    w, h = image.size
    w, h = (x - x % 64 for x in (w, h))
    image = image.resize((w, h), resample=PIL_INTERPOLATION['lanczos'])
    image = jnp.array(image).astype(dtype) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return image
