@modelkit_cli.command('predict')
@click.argument('model_name', type=str)
@click.argument('models', type=str, nargs=-1, required=False)
def predict(model_name, models):
    """
    Make predictions for a given model.
    """
    lib = _configure_from_cli_arguments(models, [model_name], {})
    model = lib.get(model_name)
    while True:
        r = click.prompt(f'[{model_name}]>')
        if r:
            res = model(json.loads(r))
            click.secho(json.dumps(res, indent=2, default=safe_np_dump))
