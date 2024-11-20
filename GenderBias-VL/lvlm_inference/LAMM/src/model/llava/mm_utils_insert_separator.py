def insert_separator(X, sep):
    return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]
