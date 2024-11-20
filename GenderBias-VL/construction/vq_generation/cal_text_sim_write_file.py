def write_file(file_name, data_to_out, fieldnames):
    f_out = open(file_name, 'w', newline='')
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    for row in data_to_out:
        writer.writerow(row)
