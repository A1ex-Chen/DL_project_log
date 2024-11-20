def add_assets_to_artifact(self, artifact, path, asset_path, split):
    img_paths = sorted(glob.glob(f'{asset_path}/*'))
    label_paths = img2label_paths(img_paths)
    for image_file, label_file in zip(img_paths, label_paths):
        image_logical_path, label_logical_path = map(lambda x: os.path.
            relpath(x, path), [image_file, label_file])
        try:
            artifact.add(image_file, logical_path=image_logical_path,
                metadata={'split': split})
            artifact.add(label_file, logical_path=label_logical_path,
                metadata={'split': split})
        except ValueError as e:
            logger.error(
                'COMET ERROR: Error adding file to Artifact. Skipping file.')
            logger.error(f'COMET ERROR: {e}')
            continue
    return artifact
