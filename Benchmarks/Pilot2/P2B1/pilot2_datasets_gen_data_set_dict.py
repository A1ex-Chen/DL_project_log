def gen_data_set_dict():
    names = {'x': 0, 'y': 1, 'z': 2, 'CHOL': 3, 'DPPC': 4, 'DIPC': 5,
        'Head': 6, 'Tail': 7}
    for i in range(12):
        temp = 'BL' + str(i + 1)
        names.update({temp: i + 8})
    fields = OrderedDict(sorted(names.items(), key=lambda t: t[1]))
    return fields
