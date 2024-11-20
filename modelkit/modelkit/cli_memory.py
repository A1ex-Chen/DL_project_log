@modelkit_cli.command()
@click.argument('models', type=str, nargs=-1, required=False)
@click.option('--required-models', '-r', multiple=True)
def memory(models, required_models):
    """
    Show memory consumption of modelkit models.
    """
    from memory_profiler import memory_usage

    def _load_model(m, service):
        service._load(m)
        sleep(1)
    service = _configure_from_cli_arguments(models, required_models, {
        'lazy_loading': True})
    grand_total = 0
    stats = {}
    logging.getLogger().setLevel(logging.ERROR)
    if service.required_models:
        with Progress(transient=True) as progress:
            task = progress.add_task('Profiling memory...', total=len(
                required_models))
            for m in service.required_models:
                deps = service.configuration[m].model_dependencies
                deps = deps.values() if isinstance(deps, dict) else deps
                for dependency in (list(deps) + [m]):
                    mu = memory_usage((_load_model, (dependency, service), {}))
                    stats[dependency] = mu[-1] - mu[0]
                    grand_total += mu[-1] - mu[0]
                progress.update(task, advance=1)
    console = Console()
    table = Table(show_header=True, header_style='bold')
    table.add_column('Model')
    table.add_column('Memory', style='dim')
    for k, (m, mc) in enumerate(stats.items()):
        table.add_row(m, humanize.naturalsize(mc * 10 ** 6, format='%.2f'),
            end_section=k == len(stats) - 1)
    table.add_row('Total', humanize.naturalsize(grand_total * 10 ** 6,
        format='%.2f'))
    console.print(table)
