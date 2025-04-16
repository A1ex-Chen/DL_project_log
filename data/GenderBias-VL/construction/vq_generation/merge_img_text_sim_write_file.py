def write_file(file_name, data_to_out, fieldnames):
    sub_dir = 'similarity'
    os.makedirs(sub_dir, exist_ok=True)
    file_name = os.path.join(sub_dir, file_name)
    f_out = open(file_name, 'w', newline='')
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    for row in data_to_out:
        writer.writerow(row)
