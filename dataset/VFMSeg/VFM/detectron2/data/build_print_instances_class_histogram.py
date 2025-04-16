def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry['annotations']
        classes = np.asarray([x['category_id'] for x in annos if not x.get(
            'iscrowd', 0)], dtype=np.int)
        if len(classes):
            assert classes.min(
                ) >= 0, f'Got an invalid category_id={classes.min()}'
            assert classes.max(
                ) < num_classes, f'Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes'
        histogram += np.histogram(classes, bins=hist_bins)[0]
    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        if len(x) > 13:
            return x[:11] + '..'
        return x
    data = list(itertools.chain(*[[short_name(class_names[i]), int(v)] for 
        i, v in enumerate(histogram)]))
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - len(data) % N_COLS))
    if num_classes > 1:
        data.extend(['total', total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(data, headers=['category', '#instances'] * (N_COLS // 
        2), tablefmt='pipe', numalign='left', stralign='center')
    log_first_n(logging.INFO, 
        'Distribution of instances among all {} categories:\n'.format(
        num_classes) + colored(table, 'cyan'), key='message')
