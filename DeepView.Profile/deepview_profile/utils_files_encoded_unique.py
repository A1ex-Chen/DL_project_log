def files_encoded_unique(operation_tree):
    encoded_files = []
    for analysis in operation_tree:
        context_info_map = analysis['operation'].get('contextInfoMap', None)
        if context_info_map is not None and len(context_info_map) > 0:
            filename = list(context_info_map[0]['context']['filePath'][
                'components']).pop()
            already_in_list = next((item for item in encoded_files if item[
                'name'] == filename), None)
            if not already_in_list:
                file_path = os.path.join('', *list(context_info_map[0][
                    'context']['filePath']['components']))
                encoded_file = encode_file('', file_path)
                encoded_files.append(encoded_file)
    return encoded_files
