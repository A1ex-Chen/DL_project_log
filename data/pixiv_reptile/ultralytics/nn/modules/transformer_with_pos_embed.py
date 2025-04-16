@staticmethod
def with_pos_embed(tensor, pos):
    """Add positional embeddings to the input tensor, if provided."""
    return tensor if pos is None else tensor + pos
