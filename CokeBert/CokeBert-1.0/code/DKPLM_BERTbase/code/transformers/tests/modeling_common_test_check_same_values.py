def check_same_values(layer_1, layer_2):
    equal = True
    for p1, p2 in zip(layer_1.weight, layer_2.weight):
        if p1.data.ne(p2.data).sum() > 0:
            equal = False
    return equal
