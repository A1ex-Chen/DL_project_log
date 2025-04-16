@staticmethod
def get_char_lens(data_file):
    return [len(x) for x in Path(data_file).open().readlines()]
