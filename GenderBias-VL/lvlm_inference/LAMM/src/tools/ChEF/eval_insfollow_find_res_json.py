def find_res_json(directory, dataset_name):
    scienceqa_files = []
    if not os.path.isdir(directory):
        return directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(dataset_name) and file.endswith('.json'):
                file_path = os.path.join(root, file)
                scienceqa_files.append(file_path)
    return scienceqa_files[0]
