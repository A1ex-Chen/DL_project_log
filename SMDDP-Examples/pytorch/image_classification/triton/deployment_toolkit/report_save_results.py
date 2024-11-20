def save_results(filename: str, data: List, formatted: bool=False):
    data = format_data(data=data) if formatted else data
    with open(filename, 'a') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
