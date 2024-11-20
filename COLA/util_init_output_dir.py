def init_output_dir(training_args):
    if training_args.do_train and os.path.exists(training_args.output_dir):
        if os.path.exists(os.path.join(training_args.output_dir,
            'checkpoint_finish')) > 0:
            raise ValueError(
                'training process in dir {} is finished, plz clear it manually.'
                .format(training_args.output_dir))
        shutil.rmtree(training_args.output_dir, ignore_errors=True)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    file_name = getattr(training_args, 'log_file', 'train.log')
    log_dir_path = os.path.join(training_args.output_dir, 'log')
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    os.system('touch {}'.format(os.path.join(training_args.output_dir,
        file_name)))
