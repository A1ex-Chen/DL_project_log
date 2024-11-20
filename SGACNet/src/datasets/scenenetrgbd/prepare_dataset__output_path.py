def _output_path(render_path, type_dir):
    path = os.path.join(args.output_path, split, type_dir, render_path)
    os.makedirs(path, exist_ok=True)
    return path
