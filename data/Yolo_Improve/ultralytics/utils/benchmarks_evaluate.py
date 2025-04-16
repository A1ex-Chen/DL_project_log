def evaluate(self, yaml_path, val_log_file, eval_log_file, list_ind):
    """
        Model evaluation on validation results.

        Args:
            yaml_path (str): YAML file path.
            val_log_file (str): val_log_file path.
            eval_log_file (str): eval_log_file path.
            list_ind (int): Index for current dataset.
        """
    skip_symbols = ['🚀', '⚠️', '💡', '❌']
    with open(yaml_path) as stream:
        class_names = yaml.safe_load(stream)['names']
    with open(val_log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        eval_lines = []
        for line in lines:
            if any(symbol in line for symbol in skip_symbols):
                continue
            entries = line.split(' ')
            entries = list(filter(lambda val: val != '', entries))
            entries = [e.strip('\n') for e in entries]
            eval_lines.extend({'class': entries[0], 'images': entries[1],
                'targets': entries[2], 'precision': entries[3], 'recall':
                entries[4], 'map50': entries[5], 'map95': entries[6]} for e in
                entries if e in class_names or e == 'all' and '(AP)' not in
                entries and '(AR)' not in entries)
    map_val = 0.0
    if len(eval_lines) > 1:
        print("There's more dicts")
        for lst in eval_lines:
            if lst['class'] == 'all':
                map_val = lst['map50']
    else:
        print("There's only one dict res")
        map_val = [res['map50'] for res in eval_lines][0]
    with open(eval_log_file, 'a') as f:
        f.write(f'{self.ds_names[list_ind]}: {map_val}\n')
