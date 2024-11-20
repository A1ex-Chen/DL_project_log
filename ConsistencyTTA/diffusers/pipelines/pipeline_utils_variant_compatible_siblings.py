def variant_compatible_siblings(filenames, variant=None) ->Union[List[os.
    PathLike], str]:
    weight_names = [WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME,
        FLAX_WEIGHTS_NAME, ONNX_WEIGHTS_NAME, ONNX_EXTERNAL_WEIGHTS_NAME]
    if is_transformers_available():
        weight_names += [TRANSFORMERS_WEIGHTS_NAME,
            TRANSFORMERS_SAFE_WEIGHTS_NAME, TRANSFORMERS_FLAX_WEIGHTS_NAME]
    weight_prefixes = [w.split('.')[0] for w in weight_names]
    weight_suffixs = [w.split('.')[-1] for w in weight_names]
    variant_file_regex = re.compile(
        f"({'|'.join(weight_prefixes)})(.{variant}.)({'|'.join(weight_suffixs)})"
        ) if variant is not None else None
    non_variant_file_regex = re.compile(f"{'|'.join(weight_names)}")
    if variant is not None:
        variant_filenames = {f for f in filenames if variant_file_regex.
            match(f.split('/')[-1]) is not None}
    else:
        variant_filenames = set()
    non_variant_filenames = {f for f in filenames if non_variant_file_regex
        .match(f.split('/')[-1]) is not None}
    usable_filenames = set(variant_filenames)
    for f in non_variant_filenames:
        variant_filename = f"{f.split('.')[0]}.{variant}.{f.split('.')[1]}"
        if variant_filename not in usable_filenames:
            usable_filenames.add(f)
    return usable_filenames, variant_filenames
