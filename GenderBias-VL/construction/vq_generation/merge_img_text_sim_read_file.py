def read_file(path):
    data, data_map = [], {}
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
            key = row['job_male'], row['job_female']
            data_map[key] = row
    return data, data_map
