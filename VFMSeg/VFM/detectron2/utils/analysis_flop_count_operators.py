def flop_count_operators(model: nn.Module, inputs: list) ->typing.DefaultDict[
    str, float]:
    """
    Implement operator-level flops counting using jit.
    This is a wrapper of :func:`fvcore.nn.flop_count` and adds supports for standard
    detection models in detectron2.
    Please use :class:`FlopCountAnalysis` for more advanced functionalities.

    Note:
        The function runs the input through the model to compute flops.
        The flops of a detection model is often input-dependent, for example,
        the flops of box & mask head depends on the number of proposals &
        the number of detected objects.
        Therefore, the flops counting using a single input may not accurately
        reflect the computation cost of a model. It's recommended to average
        across a number of inputs.

    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
            Only "image" key will be used.
        supported_ops (dict[str, Handle]): see documentation of :func:`fvcore.nn.flop_count`

    Returns:
        Counter: Gflop count per operator
    """
    old_train = model.training
    model.eval()
    ret = FlopCountAnalysis(model, inputs).by_operator()
    model.train(old_train)
    return {k: (v / 1000000000.0) for k, v in ret.items()}
