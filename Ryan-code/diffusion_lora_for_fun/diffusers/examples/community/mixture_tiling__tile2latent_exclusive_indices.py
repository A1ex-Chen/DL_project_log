def _tile2latent_exclusive_indices(tile_row, tile_col, tile_width,
    tile_height, tile_row_overlap, tile_col_overlap, rows, columns):
    """Given a tile row and column numbers returns the range of latents affected only by that tile in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    row_init, row_end, col_init, col_end = _tile2latent_indices(tile_row,
        tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap)
    row_segment = segment(row_init, row_end)
    col_segment = segment(col_init, col_end)
    for row in range(rows):
        for column in range(columns):
            if row != tile_row and column != tile_col:
                (clip_row_init, clip_row_end, clip_col_init, clip_col_end) = (
                    _tile2latent_indices(row, column, tile_width,
                    tile_height, tile_row_overlap, tile_col_overlap))
                row_segment = row_segment - segment(clip_row_init, clip_row_end
                    )
                col_segment = col_segment - segment(clip_col_init, clip_col_end
                    )
    return row_segment[0], row_segment[1], col_segment[0], col_segment[1]
