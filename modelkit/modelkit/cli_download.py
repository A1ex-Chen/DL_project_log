@modelkit_cli.command('download-assets')
@click.argument('models', type=str, nargs=-1, required=False)
@click.option('--required-models', '-r', multiple=True)
def download(models, required_models):
    """
    Download all assets necessary to run a given set of models
    """
    download_assets(models=list(models) or None, required_models=list(
        required_models) or None)
