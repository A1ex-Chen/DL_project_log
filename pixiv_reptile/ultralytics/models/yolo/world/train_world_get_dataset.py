def get_dataset(self):
    """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
    final_data = {}
    data_yaml = self.args.data
    assert data_yaml.get('train', False), 'train dataset not found'
    assert data_yaml.get('val', False), 'validation dataset not found'
    data = {k: [check_det_dataset(d) for d in v.get('yolo_data', [])] for k,
        v in data_yaml.items()}
    assert len(data['val']
        ) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
    val_split = 'minival' if 'lvis' in data['val'][0]['val'] else 'val'
    for d in data['val']:
        if d.get('minival') is None:
            continue
        d['minival'] = str(d['path'] / d['minival'])
    for s in ['train', 'val']:
        final_data[s] = [d['train' if s == 'train' else val_split] for d in
            data[s]]
        grounding_data = data_yaml[s].get('grounding_data')
        if grounding_data is None:
            continue
        grounding_data = grounding_data if isinstance(grounding_data, list
            ) else [grounding_data]
        for g in grounding_data:
            assert isinstance(g, dict
                ), f'Grounding data should be provided in dict format, but got {type(g)}'
        final_data[s] += grounding_data
    final_data['nc'] = data['val'][0]['nc']
    final_data['names'] = data['val'][0]['names']
    self.data = final_data
    return final_data['train'], final_data['val'][0]
