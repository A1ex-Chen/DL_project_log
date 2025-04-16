@modelkit_cli.command('tf-serving')
@click.argument('mode', type=click.Choice(['local-docker', 'local-process',
    'remote']))
@click.argument('models', type=str, nargs=-1, required=False)
@click.option('--required-models', '-r', multiple=True)
@click.option('--verbose', is_flag=True)
def tf_serving(mode, models, required_models, verbose):
    from modelkit.utils.tensorflow import deploy_tf_models
    service = _configure_from_cli_arguments(models, required_models, {
        'lazy_loading': True})
    deploy_tf_models(service, mode, verbose=verbose)
