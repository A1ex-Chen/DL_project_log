@modelkit_cli.command()
@click.argument('models', type=str, nargs=-1, required=False)
@click.option('--required-models', '-r', multiple=True)
def describe(models, required_models):
    """
    Describe a library.

    Show settings, models and assets for a given library.
    """
    service = _configure_from_cli_arguments(models, required_models, {})
    service.describe()
