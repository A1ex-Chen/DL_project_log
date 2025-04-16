def write_csv(self, file_name, data, fieldnames=None):
    write_data = copy.deepcopy(data)
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, mode='w', newline='') as file:
        if fieldnames is None:
            writer = csv.DictWriter(file, fieldnames=list(write_data[0].keys())
                )
        else:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in write_data:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    row[key] = f'{value:.2f}'
            writer.writerow(row)
