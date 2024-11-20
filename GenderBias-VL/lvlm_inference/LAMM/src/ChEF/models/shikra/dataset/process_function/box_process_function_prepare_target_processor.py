def prepare_target_processor(model, preprocessor: Dict[str, Any],
    model_args, training_args):
    if not hasattr(model_args, 'target_processor'):
        return model, preprocessor
    target_processor = {}
    if 'boxes' in model_args['target_processor']:
        boxes_cfg = model_args['target_processor']['boxes']
        boxes_processor = BOXES_PROCESSOR.build(boxes_cfg)
        target_processor['boxes'] = boxes_processor
        if hasattr(boxes_processor, 'post_process_model_tokenizer'):
            model, preprocessor = boxes_processor.post_process_model_tokenizer(
                model, preprocessor, model_args, training_args)
    preprocessor['target'] = target_processor
    return model, preprocessor
