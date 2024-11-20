def setup_to_resume(args, model, optimizer):
    chk_dict = torch.load(os.path.join(os.path.abspath(args.resume_training
        ), 'models/checkpoint-recent.pth'))
    model.load_state_dict(chk_dict[STATE_DICT_KEY])
    optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])
