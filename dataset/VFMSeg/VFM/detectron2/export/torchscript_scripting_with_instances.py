def scripting_with_instances(model, fields):
    """
    Run :func:`torch.jit.script` on a model that uses the :class:`Instances` class. Since
    attributes of :class:`Instances` are "dynamically" added in eager mode，it is difficult
    for scripting to support it out of the box. This function is made to support scripting
    a model that uses :class:`Instances`. It does the following:

    1. Create a scriptable ``new_Instances`` class which behaves similarly to ``Instances``,
       but with all attributes been "static".
       The attributes need to be statically declared in the ``fields`` argument.
    2. Register ``new_Instances``, and force scripting compiler to
       use it when trying to compile ``Instances``.

    After this function, the process will be reverted. User should be able to script another model
    using different fields.

    Example:
        Assume that ``Instances`` in the model consist of two attributes named
        ``proposal_boxes`` and ``objectness_logits`` with type :class:`Boxes` and
        :class:`Tensor` respectively during inference. You can call this function like:
        ::
            fields = {"proposal_boxes": Boxes, "objectness_logits": torch.Tensor}
            torchscipt_model =  scripting_with_instances(model, fields)

    Note:
        It only support models in evaluation mode.

    Args:
        model (nn.Module): The input model to be exported by scripting.
        fields (Dict[str, type]): Attribute names and corresponding type that
            ``Instances`` will use in the model. Note that all attributes used in ``Instances``
            need to be added, regardless of whether they are inputs/outputs of the model.
            Data type not defined in detectron2 is not supported for now.

    Returns:
        torch.jit.ScriptModule: the model in torchscript format
    """
    assert not model.training, 'Currently we only support exporting models in evaluation mode to torchscript'
    with freeze_training_mode(model), patch_instances(fields):
        scripted_model = torch.jit.script(model)
        return scripted_model
