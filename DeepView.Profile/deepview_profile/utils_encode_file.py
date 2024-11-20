def encode_file(root, file):
    file_dict = None
    if os.path.splitext(file)[1] == '.py' and file != 'entry_point.py':
        file_dict = {'name': file, 'content': ''}
        filename = os.path.join(root, file)
        with open(filename, 'r') as f:
            file_content = f.read()
            file_dict['content'] = base64.b64encode(file_content.encode(
                'utf-8')).decode('utf-8')
    return file_dict
