def infer_option(self, answer):

    def get_unit_option(splits, choices='ABCD', prefix='', suffix=''):
        res = []
        for c in choices:
            if prefix + c + suffix in splits:
                res.append(c)
        return res
    splits = [x.strip() for x in answer.split()]
    no_prefix_option = get_unit_option(splits, self.choices)
    if len(no_prefix_option) == 1:
        if 'A' not in splits or len(splits) < 3:
            return no_prefix_option[0]
    tups = [('(', ')'), ('(', ').'), ('', '.'), ('', ','), ('', ':'), ('',
        ')'), ('', ').'), (':', ''), (':', ','), (':', '.'), (':', ')'), (
        ':', ').')]
    for tup in tups:
        prefix_option = get_unit_option(splits, self.choices, prefix=tup[0],
            suffix=tup[1])
        if len(prefix_option) == 1:
            return prefix_option[0]
    return None
