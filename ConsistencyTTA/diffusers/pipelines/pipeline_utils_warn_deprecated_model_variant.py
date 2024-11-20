def warn_deprecated_model_variant(pretrained_model_name_or_path,
    use_auth_token, variant, revision, model_filenames):
    info = model_info(pretrained_model_name_or_path, use_auth_token=
        use_auth_token, revision=None)
    filenames = {sibling.rfilename for sibling in info.siblings}
    comp_model_filenames, _ = variant_compatible_siblings(filenames,
        variant=revision)
    comp_model_filenames = ['.'.join(f.split('.')[:1] + f.split('.')[2:]) for
        f in comp_model_filenames]
    if set(comp_model_filenames) == set(model_filenames):
        warnings.warn(
            f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` even though you can load it via `variant=`{revision}`. Loading model variants via `revision='{revision}'` is deprecated and will be removed in diffusers v1. Please use `variant='{revision}'` instead."
            , FutureWarning)
    else:
        warnings.warn(
            f"""You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have the required variant filenames in the 'main' branch. 
 The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {revision} files' so that the correct variant file can be added."""
            , FutureWarning)
