def write_csv(file_name, data, mode='w', fieldnames=None):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    file_flag = os.path.exists(file_name)
    with open(file_name, mode, newline='') as file:
        if fieldnames is None:
            writer = csv.DictWriter(file, fieldnames=list(data[0].keys()))
        else:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    row[key] = f'{value:.2f}'
            writer.writerow(row)
