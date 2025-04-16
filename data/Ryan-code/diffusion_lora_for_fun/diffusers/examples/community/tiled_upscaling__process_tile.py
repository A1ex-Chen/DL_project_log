def _process_tile(self, original_image_slice, x, y, tile_size, tile_border,
    image, final_image, **kwargs):
    torch.manual_seed(0)
    crop_rect = min(image.size[0] - (tile_size + original_image_slice), x *
        tile_size), min(image.size[1] - (tile_size + original_image_slice),
        y * tile_size), min(image.size[0], (x + 1) * tile_size), min(image.
        size[1], (y + 1) * tile_size)
    crop_rect_with_overlap = add_overlap_rect(crop_rect, tile_border, image
        .size)
    tile = image.crop(crop_rect_with_overlap)
    translated_slice_x = (crop_rect[0] + (crop_rect[2] - crop_rect[0]) / 2
        ) / image.size[0] * tile.size[0]
    translated_slice_x = translated_slice_x - original_image_slice / 2
    translated_slice_x = max(0, translated_slice_x)
    to_input = squeeze_tile(tile, image, original_image_slice,
        translated_slice_x)
    orig_input_size = to_input.size
    to_input = to_input.resize((tile_size, tile_size), Image.BICUBIC)
    upscaled_tile = super(StableDiffusionTiledUpscalePipeline, self).__call__(
        image=to_input, **kwargs).images[0]
    upscaled_tile = upscaled_tile.resize((orig_input_size[0] * 4, 
        orig_input_size[1] * 4), Image.BICUBIC)
    upscaled_tile = unsqueeze_tile(upscaled_tile, original_image_slice)
    upscaled_tile = upscaled_tile.resize((tile.size[0] * 4, tile.size[1] * 
        4), Image.BICUBIC)
    remove_borders = []
    if x == 0:
        remove_borders.append('l')
    elif crop_rect[2] == image.size[0]:
        remove_borders.append('r')
    if y == 0:
        remove_borders.append('t')
    elif crop_rect[3] == image.size[1]:
        remove_borders.append('b')
    transparency_mask = Image.fromarray(make_transparency_mask((
        upscaled_tile.size[0], upscaled_tile.size[1]), tile_border * 4,
        remove_borders=remove_borders), mode='L')
    final_image.paste(upscaled_tile, (crop_rect_with_overlap[0] * 4, 
        crop_rect_with_overlap[1] * 4), transparency_mask)
