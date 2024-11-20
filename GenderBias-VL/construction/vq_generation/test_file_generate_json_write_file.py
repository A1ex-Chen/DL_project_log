def write_file(file_name, data_to_out):
    with open(file_name, 'w', encoding='utf8') as f:
        f.write(json.dumps(data_to_out, indent=4, ensure_ascii=False))
