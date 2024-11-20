@staticmethod
def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos
