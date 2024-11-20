def get_dataset(self):
    """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
    try:
        if self.args.task == 'classify':
            data = check_cls_dataset(self.args.data)
        elif self.args.data.split('.')[-1] in {'yaml', 'yml'
            } or self.args.task in {'detect', 'segment', 'pose', 'obb'}:
            data = check_det_dataset(self.args.data)
            if 'yaml_file' in data:
                self.args.data = data['yaml_file']
    except Exception as e:
        raise RuntimeError(emojis(
            f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
    self.data = data
    return data['train'], data.get('val') or data.get('create_self_data')
