@modelkit_cli.command('batch')
@click.argument('model_name', type=str)
@click.argument('input', type=str)
@click.argument('output', type=str)
@click.option('--models', type=str, multiple=True)
@click.option('--processes', type=int, default=None)
@click.option('--unordered', is_flag=True)
def batch_predict(model_name, input, output, models, processes, unordered):
    """
    Barch predictions for a given model.
    """
    processes = processes or os.cpu_count()
    print(f'Using {processes} processes')
    lib = _configure_from_cli_arguments(models, [model_name], {
        'lazy_loading': True})
    manager = multiprocessing.Manager()
    results_queue = manager.Queue()
    n_workers = processes - 2
    items_queues = [manager.Queue() for _ in range(n_workers)]
    with multiprocessing.Pool(processes) as p:
        workers = [p.apply_async(worker, (lib, model_name, q_in,
            results_queue)) for q_in in items_queues]
        p.apply_async(reader, (input, items_queues))
        if unordered:
            r = p.apply_async(writer_unordered, (output, results_queue,
                n_workers))
        else:
            r = p.apply_async(writer, (output, results_queue, n_workers))
        wrote_items = r.get()
        for k, w in enumerate(workers):
            print(f'Worker {k} computed {w.get()} elements')
        print(f'Total: {wrote_items} elements')
