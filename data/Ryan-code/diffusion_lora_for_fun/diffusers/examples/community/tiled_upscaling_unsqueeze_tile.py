def unsqueeze_tile(tile, original_image_slice):
    crop_rect = original_image_slice * 4, 0, tile.size[0], tile.size[1]
    tile = tile.crop(crop_rect)
    return tile
