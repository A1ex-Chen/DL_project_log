def print_statistics(json_data):
    categories = defaultdict(int)
    licenses = defaultdict(int)
    countries = defaultdict(int)
    for item in json_data.values():
        categories[item['category']] += 1
        licenses[item['metadata']['license']] += 1
        country = item['metadata']['country']
        countries[country if country else '<Unknown>'] += 1
    for name in [name for name, num in countries.items() if num / len(
        json_data) < 0.001]:
        countries['<Other>'] += countries.pop(name)
    rows = [[]] * 2
    rows += [['Category', 'Images', '% of all']]
    rows += [['---'] * 3]
    for name, num in sorted(categories.items(), key=lambda x: -x[1]):
        rows += [[name, '%d' % num, '%.2f' % (100.0 * num / len(json_data))]]
    rows += [[]] * 2
    rows += [['License', 'Images', '% of all']]
    rows += [['---'] * 3]
    for name, num in sorted(licenses.items(), key=lambda x: -x[1]):
        rows += [[name, '%d' % num, '%.2f' % (100.0 * num / len(json_data))]]
    rows += [[]] * 2
    rows += [['Country', 'Images', '% of all', '% of known']]
    rows += [['---'] * 4]
    for name, num in sorted(countries.items(), key=lambda x: -x[1] if x[0] !=
        '<Other>' else 0):
        rows += [[name, '%d' % num, '%.2f' % (100.0 * num / len(json_data)),
            '%.2f' % (0 if name == '<Unknown>' else 100.0 * num / (len(
            json_data) - countries['<Unknown>']))]]
    rows += [[]] * 2
    widths = [max(len(cell) for cell in column if cell is not None) for
        column in itertools.zip_longest(*rows)]
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in
            zip(row, widths)))
