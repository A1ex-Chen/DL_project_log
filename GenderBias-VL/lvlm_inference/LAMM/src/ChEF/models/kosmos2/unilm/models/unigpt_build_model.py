@classmethod
def build_model(cls, args, task):
    if hasattr(task, 'all_dict'):
        task.dictionary = task.all_dict
    if task.__class__.__name__ == 'GenerationObjTask':
        gpt_model = GPTEvalmodel.build_model(args, task)
    else:
        gpt_model = GPTmodel.build_model(args, task)
    logger.info('gpt args is'.format(args))
    text_model, text_connector = cls.load_text_model(args, task)
    img_model, img_connector = cls.load_image_model(args, task)
    aud_model, aud_connector = cls.load_audio_model(args, task)
    model = cls(args, gpt_model, text_model=text_model, text_connector=
        text_connector, img_model=img_model, img_connector=img_connector,
        aud_model=aud_model, aud_connector=aud_connector, bos=task.
        dictionary.bos_index, eos=task.dictionary.eos_index)
    if args.pretrained_ckpt_path != '':
        state = checkpoint_utils.load_checkpoint_to_cpu(args.
            pretrained_ckpt_path)
        model.load_state_dict(state['model'], strict=True, args=args)
    if model.text_model is not None:
        for p in model.text_model.parameters():
            p.requires_grad = False
    if model.img_model is not None:
        for p_name, p in model.img_model.named_parameters():
            if args.no_freeze_layer:
                no_freeze_layers = args.no_freeze_layer.split(',')
                for no_freeze_layer in no_freeze_layers:
                    if no_freeze_layer in p_name:
                        print('no_freeze_layer: {}'.format(p_name))
                        p.requires_grad = True
                        break
            p.requires_grad = False
    if model.aud_model is not None:
        for p in model.aud_model.parameters():
            p.requires_grad = False
    return model
