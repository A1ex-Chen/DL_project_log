def get_supported_pipeline_table() ->dict:
    """
    Generates a dictionary containing the supported auto classes for each pipeline type,
    using the content of the auto modules.
    """
    all_supported_pipeline_classes = [(class_name.__name__, 'text-to-image',
        'AutoPipelineForText2Image') for _, class_name in
        AUTO_TEXT2IMAGE_PIPELINES_MAPPING.items()]
    all_supported_pipeline_classes += [(class_name.__name__,
        'image-to-image', 'AutoPipelineForImage2Image') for _, class_name in
        AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.items()]
    all_supported_pipeline_classes += [(class_name.__name__,
        'image-to-image', 'AutoPipelineForInpainting') for _, class_name in
        AUTO_INPAINT_PIPELINES_MAPPING.items()]
    all_supported_pipeline_classes = list(set(all_supported_pipeline_classes))
    all_supported_pipeline_classes.sort(key=lambda x: x[0])
    data = {}
    data['pipeline_class'] = [sample[0] for sample in
        all_supported_pipeline_classes]
    data['pipeline_tag'] = [sample[1] for sample in
        all_supported_pipeline_classes]
    data['auto_class'] = [sample[2] for sample in
        all_supported_pipeline_classes]
    return data
