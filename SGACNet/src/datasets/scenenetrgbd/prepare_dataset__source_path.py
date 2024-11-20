def _source_path(render_path, type_dir):
    split_ = 'val' if split == 'valid' else split
    return os.path.join(args.scenenetrgbd_filepath, split_, render_path,
        type_dir)
