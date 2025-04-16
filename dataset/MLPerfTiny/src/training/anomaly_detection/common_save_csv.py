def save_csv(save_file_path, save_data):
    with open(save_file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)
