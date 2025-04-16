def model_has_vae(args):
    config_file_name = os.path.join('vae', AutoencoderKL.config_name)
    if os.path.isdir(args.pretrained_model_name_or_path):
        config_file_name = os.path.join(args.pretrained_model_name_or_path,
            config_file_name)
        return os.path.isfile(config_file_name)
    else:
        files_in_repo = model_info(args.pretrained_model_name_or_path,
            revision=args.revision).siblings
        return any(file.rfilename == config_file_name for file in files_in_repo
            )
