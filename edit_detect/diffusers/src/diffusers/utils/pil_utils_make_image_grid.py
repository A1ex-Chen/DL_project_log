def make_image_grid(images: List[PIL.Image.Image], rows: int, cols: int,
    resize: int=None) ->PIL.Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols
    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
