def squeeze_tile(tile, original_image, original_slice, slice_x):
    result = Image.new('RGB', (tile.size[0] + original_slice, tile.size[1]))
    result.paste(original_image.resize((tile.size[0], tile.size[1]), Image.
        BICUBIC).crop((slice_x, 0, slice_x + original_slice, tile.size[1])),
        (0, 0))
    result.paste(tile, (original_slice, 0))
    return result
