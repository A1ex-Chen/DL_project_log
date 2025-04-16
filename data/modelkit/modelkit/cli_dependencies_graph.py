@modelkit_cli.command()
@click.argument('models', type=str, nargs=-1, required=False)
@click.option('--required-models', '-r', multiple=True)
def dependencies_graph(models, required_models):
    import networkx as nx
    from networkx.drawing.nx_agraph import write_dot
    """
    Create a  dependency graph for a library.

    Create a DOT file with the assets and model dependency graph
    from a list of models.
    """
    service = _configure_from_cli_arguments(models, required_models, {
        'lazy_loading': True})
    if service.configuration:
        dependency_graph = nx.DiGraph(overlap='False')
        for m in service.required_models:
            add_dependencies_to_graph(dependency_graph, m, service.
                configuration)
        write_dot(dependency_graph, 'dependencies.dot')
