def read_file(path):
    data = []
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key, value in row.items():
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
            data.append(row)
    return data
