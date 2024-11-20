def write_csv(self, file_name, data):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0].keys()))
        writer.writeheader()
        for row in data:
            for key, value in row.items():
                if isinstance(value, (int, float)
                    ) and key != 'occtm_ratio' and key != 'occtf_ratio':
                    row[key] = f'{100 * value:.2f}'
                elif isinstance(value, (int, float)):
                    row[key] = f'{value:.2f}'
            writer.writerow(row)
