def __init__(self, args):
    self.args = args
    logfile = args.log_file
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    self.flog = open(os.path.join(exp_dir, 'logs', logfile), 'w')
    os.makedirs(os.path.join(exp_dir, 'prompt_records'), exist_ok=True)
    self.flog_propmpt = open(os.path.join(exp_dir, 'prompt_records',
        logfile), 'w')
    os.makedirs(os.path.join(fail_exp_dir, 'logs'), exist_ok=True)
    self.fail_flog = open(os.path.join(fail_exp_dir, 'logs', logfile), 'w')
    os.makedirs(os.path.join(fail_exp_dir, 'prompt_records'), exist_ok=True)
    self.fail_flog_propmpt = open(os.path.join(fail_exp_dir,
        'prompt_records', logfile), 'w')
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.setup_seed()
    self.init_all_occ()
    self.build_clip_model()
    self.build_stable_diffusion_model()
    print(f'start_index: {args.start_index}, end_index: {args.end_index}')
    self.generate_images(args.start_index, args.end_index)
