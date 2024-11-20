def build(input_reader_config, model_config, training, voxel_generator,
    target_assigner=None) ->DatasetWrapper:
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError(
            'input_reader_config not of type input_reader_pb2.InputReader.')
    dataset = dataset_builder.build(input_reader_config, model_config,
        training, voxel_generator, target_assigner)
    dataset = DatasetWrapper(dataset)
    return dataset
