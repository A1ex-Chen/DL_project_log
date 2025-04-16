@modelkit_cli.command('list-assets')
@click.argument('models', type=str, nargs=-1, required=False)
@click.option('--required-models', '-r', multiple=True)
def list_assets_cli(models, required_models):
    """
    List necessary assets.

    List the assets necessary to run a given set of models.
    """
    service = _configure_from_cli_arguments(models, required_models, {
        'lazy_loading': True})
    console = Console()
    if service.configuration:
        for m in service.required_models:
            assets_specs = list_assets(configuration=service.configuration,
                required_models=[m])
            model_tree = Tree(f'[bold]{m}[/bold] ({len(assets_specs)} assets)')
            if assets_specs:
                for asset_spec_string in assets_specs:
                    model_tree.add(escape(asset_spec_string), style='dim')
            console.print(model_tree)
