@modelkit_cli.command()
@click.argument('model')
@click.argument('example')
@click.argument('models', type=str, nargs=-1, required=False)
@click.option('--n', '-n', default=100)
def time(model, example, models, n):
    """
    Benchmark a model on an example.

    Time n iterations of a model's call on an example.
    """
    service = _configure_from_cli_arguments(models, [model], {
        'lazy_loading': True})
    console = Console()
    t0 = perf_counter()
    model = service.get(model)
    console.print(
        f"{f'Loaded model `{model.configuration_key}` in':50} ... {f'{perf_counter() - t0:.2f} s':>10}"
        )
    example_deserialized = json.loads(example)
    console.print(f'Calling `predict` {n} times on example:')
    console.print(f'{json.dumps(example_deserialized, indent=2)}')
    times = []
    for _ in track(range(n)):
        t0 = perf_counter()
        model(example_deserialized)
        times.append(perf_counter() - t0)
    console.print(
        f'Finished in {sum(times):.1f} s, approximately {sum(times) / n * 1000.0:.2f} ms per call'
        )
    t0 = perf_counter()
    model([example_deserialized] * n)
    batch_time = perf_counter() - t0
    console.print(
        f'Finished batching in {batch_time:.1f} s, approximately {batch_time / n * 1000.0:.2f} ms per call'
        )
