@modelkit_cli.command('serve')
@click.argument('models', type=str, nargs=-1, required=False)
@click.option('--required-models', '-r', type=str, multiple=True)
@click.option('--host', type=str, default='localhost')
@click.option('--port', type=int, default=8000)
def serve(models, required_models, host, port):
    import uvicorn
    """
    Run a library as a service.

    Run an HTTP server with specified models using FastAPI
    """
    app = create_modelkit_app(models=list(models) or None, required_models=
        list(required_models) or None)
    uvicorn.run(app, host=host, port=port)
