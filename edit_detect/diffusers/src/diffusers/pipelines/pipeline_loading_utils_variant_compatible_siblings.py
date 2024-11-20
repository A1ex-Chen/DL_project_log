def variant_compatible_siblings(filenames, variant=None) ->Union[List[os.
    PathLike], str]:
    weight_names = [WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME,
        FLAX_WEIGHTS_NAME, ONNX_WEIGHTS_NAME, ONNX_EXTERNAL_WEIGHTS_NAME]
    if is_transformers_available():
        weight_names += [TRANSFORMERS_WEIGHTS_NAME,
            TRANSFORMERS_SAFE_WEIGHTS_NAME, TRANSFORMERS_FLAX_WEIGHTS_NAME]
    weight_prefixes = [w.split('.')[0] for w in weight_names]
    weight_suffixs = [w.split('.')[-1] for w in weight_names]
    transformers_index_format = '\\d{5}-of-\\d{5}'
    if variant is not None:
        variant_file_re = re.compile(
            f"({'|'.join(weight_prefixes)})\\.({variant}|{variant}-{transformers_index_format})\\.({'|'.join(weight_suffixs)})$"
            )
        variant_index_re = re.compile(
            f"({'|'.join(weight_prefixes)})\\.({'|'.join(weight_suffixs)})\\.index\\.{variant}\\.json$"
            )
    non_variant_file_re = re.compile(
        f"({'|'.join(weight_prefixes)})(-{transformers_index_format})?\\.({'|'.join(weight_suffixs)})$"
        )
    non_variant_index_re = re.compile(
        f"({'|'.join(weight_prefixes)})\\.({'|'.join(weight_suffixs)})\\.index\\.json"
        )
    if variant is not None:
        variant_weights = {f for f in filenames if variant_file_re.match(f.
            split('/')[-1]) is not None}
        variant_indexes = {f for f in filenames if variant_index_re.match(f
            .split('/')[-1]) is not None}
        variant_filenames = variant_weights | variant_indexes
    else:
        variant_filenames = set()
    non_variant_weights = {f for f in filenames if non_variant_file_re.
        match(f.split('/')[-1]) is not None}
    non_variant_indexes = {f for f in filenames if non_variant_index_re.
        match(f.split('/')[-1]) is not None}
    non_variant_filenames = non_variant_weights | non_variant_indexes
    usable_filenames = set(variant_filenames)

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
    for f in non_variant_filenames:
        variant_filename = convert_to_variant(f)
        if variant_filename not in usable_filenames:
            usable_filenames.add(f)
    return usable_filenames, variant_filenames
