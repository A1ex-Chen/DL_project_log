def read_occ_file(self, path):
    data = []
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row['occupation'])
    return data
