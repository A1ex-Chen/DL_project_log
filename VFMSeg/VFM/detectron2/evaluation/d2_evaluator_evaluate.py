def evaluate(self):
    results = super().evaluate()
    if results is None:
        return
    results_per_category = []
    for name in self._class_names:
        results_per_category.append((str(name), float(results['sem_seg'][
            f'IoU-{name}'])))
    N_COLS = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in
        range(N_COLS)])
    table = tabulate(results_2d, tablefmt='pipe', floatfmt='.3f', headers=[
        'category', 'IoU'] * (N_COLS // 2), numalign='left')
    self._logger.info('Per-category IoU: \n' + table)
    prefix_results = OrderedDict()
    for k, v in results.items():
        prefix_results[f'{self.dataset_name}/{self.prefix}{k}'] = v
    return prefix_results
