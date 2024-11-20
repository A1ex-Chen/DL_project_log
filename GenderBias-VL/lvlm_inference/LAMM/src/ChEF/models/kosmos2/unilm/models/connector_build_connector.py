def build_connector(args, input_dim, output_dim):
    if isinstance(args, str):
        connector_name = args
    else:
        connector_name = args.text_connector if hasattr(args, 'text_connector'
            ) else args.connector
    if connector_name == 'none':
        connector = None
    elif connector_name == 'complex':
        connector = ComplexConnector(input_dim, output_dim, args.activation_fn)
    elif connector_name == 'simple':
        connector = SimpleConnector(input_dim, output_dim)
    elif connector_name == 'xconnector':
        connector = XConnector(input_dim, output_dim, args)
    else:
        raise ValueError('Invalid text connector type: {}'.format(
            connector_name))
    return connector
