def add_dependencies_to_graph(g, model, configurations):
    g.add_node(model, type='model', fillcolor='/accent3/2', style='filled',
        shape='box')
    model_configuration = configurations[model]
    if model_configuration.asset:
        g.add_node(model_configuration.asset, type='asset', fillcolor=
            '/accent3/3', style='filled')
        g.add_edge(model, model_configuration.asset)
    for dependent_model in model_configuration.model_dependencies:
        g.add_edge(model, dependent_model)
        add_dependencies_to_graph(g, dependent_model, configurations)
