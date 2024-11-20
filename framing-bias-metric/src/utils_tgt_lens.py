@cached_property
def tgt_lens(self):
    """Length in characters of target documents"""
    return self.get_char_lens(self.tgt_file)
