def find_duplicates(seq):
    return [k for k, v in collections.Counter(seq).items() if v > 1]
