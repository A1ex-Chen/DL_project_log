def convert_to_sent(self, indices, exclude=None):
    if exclude is None:
        return ' '.join([self.get_sym(idx) for idx in indices])
    else:
        return ' '.join([self.get_sym(idx) for idx in indices if idx not in
            exclude])
