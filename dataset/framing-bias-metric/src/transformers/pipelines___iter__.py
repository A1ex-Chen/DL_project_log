def __iter__(self):
    for line in sys.stdin:
        if '\t' in line:
            line = line.split('\t')
            if self.column:
                yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}
            else:
                yield tuple(line)
        else:
            yield line
