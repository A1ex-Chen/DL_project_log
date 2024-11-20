@contextmanager
def patch_builtin_len(modules=()):
    """
    Patch the builtin len() function of a few detectron2 modules
    to use __len__ instead, because __len__ does not convert values to
    integers and therefore is friendly to tracing.

    Args:
        modules (list[stsr]): names of extra modules to patch len(), in
            addition to those in detectron2.
    """

    def _new_len(obj):
        return obj.__len__()
    with ExitStack() as stack:
        MODULES = ['detectron2.modeling.roi_heads.fast_rcnn',
            'detectron2.modeling.roi_heads.mask_head',
            'detectron2.modeling.roi_heads.keypoint_head'] + list(modules)
        ctxs = [stack.enter_context(mock.patch(mod + '.len')) for mod in
            MODULES]
        for m in ctxs:
            m.side_effect = _new_len
        yield
