def get_ann_path(idx, *, annotation_dir=''):
    return os.path.join(annotation_dir, f'Annotations/{idx}.xml')
