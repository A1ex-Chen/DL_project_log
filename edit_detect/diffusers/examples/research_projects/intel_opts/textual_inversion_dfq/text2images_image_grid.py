def image_grid(imgs, rows, cols):
    if not len(imgs) == rows * cols:
        raise ValueError(
            'The specified number of rows and columns are not correct.')
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
