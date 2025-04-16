def _tile2latent_indices(tile_row, tile_col, tile_width, tile_height,
    tile_row_overlap, tile_col_overlap):
    """Given a tile row and column numbers returns the range of latents affected by that tiles in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(
        tile_row, tile_col, tile_width, tile_height, tile_row_overlap,
        tile_col_overlap)
    return _pixel2latent_indices(px_row_init, px_row_end, px_col_init,
        px_col_end)
