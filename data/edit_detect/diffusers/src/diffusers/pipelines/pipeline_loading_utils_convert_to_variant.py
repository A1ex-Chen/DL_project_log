def convert_to_variant(filename):
    if 'index' in filename:
        variant_filename = filename.replace('index', f'index.{variant}')
    elif re.compile(f'^(.*?){transformers_index_format}').match(filename
        ) is not None:
        variant_filename = (
            f"{filename.split('-')[0]}.{variant}-{'-'.join(filename.split('-')[1:])}"
            )
    else:
        variant_filename = (
            f"{filename.split('.')[0]}.{variant}.{filename.split('.')[1]}")
    return variant_filename
