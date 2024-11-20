def save_model_graph(self, image_shape: Size):
    graph = Digraph()
    graph.node(name='image', label=f'Image\n{str(tuple(image_shape))}',
        shape='box', style='filled', fillcolor='#ffffff')
    try:
        subgraph, input_node_name, output_node_name = (self.model.algorithm
            .make_graph())
        graph.subgraph(subgraph)
    except NotImplementedError:
        with graph.subgraph(name='cluster_model') as c:
            c.attr(label=f'Model', style='filled', color='lightgray')
            c.node(name='net', label=f'{self.config.algorithm_name.value}',
                shape='box', style='filled', fillcolor='#ffffff')
            input_node_name = output_node_name = 'net'
    graph.node(name='output', label=
        f'Output\n{str((self.model.num_classes,))}', shape='box', style=
        'filled', fillcolor='#ffffff')
    graph.edge('image', input_node_name)
    graph.edge(output_node_name, 'output')
    graph.render(filename='model-graph', directory=self.
        path_to_checkpoints_dir, format='png', cleanup=True)
