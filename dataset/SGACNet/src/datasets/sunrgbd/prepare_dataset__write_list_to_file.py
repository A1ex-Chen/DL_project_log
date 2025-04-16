def _write_list_to_file(list_, filepath):
    with open(os.path.join(output_path, filepath), 'w') as f:
        f.write('\n'.join(list_))
    print('written file {}'.format(filepath))
