def _pixel2latent_indices(px_row_init, px_row_end, px_col_init, px_col_end):
    """Translates coordinates in pixel space to coordinates in latent space"""
    return px_row_init // 8, px_row_end // 8, px_col_init // 8, px_col_end // 8
