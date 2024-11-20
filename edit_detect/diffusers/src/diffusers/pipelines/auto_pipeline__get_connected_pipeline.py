def _get_connected_pipeline(pipeline_cls):
    if pipeline_cls in _AUTO_TEXT2IMAGE_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
            pipeline_cls.__name__, throw_error_if_not_exist=False)
    if pipeline_cls in _AUTO_IMAGE2IMAGE_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
            pipeline_cls.__name__, throw_error_if_not_exist=False)
    if pipeline_cls in _AUTO_INPAINT_DECODER_PIPELINES_MAPPING.values():
        return _get_task_class(AUTO_INPAINT_PIPELINES_MAPPING, pipeline_cls
            .__name__, throw_error_if_not_exist=False)
