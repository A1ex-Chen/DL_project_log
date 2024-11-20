def activation_count_operators(model: nn.Module, inputs: list, **kwargs
    ) ->typing.DefaultDict[str, float]:
    """
    Implement operator-level activations counting using jit.
    This is a wrapper of fvcore.nn.activation_count, that supports standard detection models
    in detectron2.

    Note:
        The function runs the input through the model to compute activations.
        The activations of a detection model is often input-dependent, for example,
        the activations of box & mask head depends on the number of proposals &
        the number of detected objects.

    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
            Only "image" key will be used.

    Returns:
        Counter: activation count per operator
    """
    return _wrapper_count_operators(model=model, inputs=inputs, mode=
        ACTIVATIONS_MODE, **kwargs)
