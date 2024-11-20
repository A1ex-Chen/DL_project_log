def create_runner_config_validator():
    validator = ConfigValidator(description='Runner configurations')
    validator.add_argument('runner', type=str, choices=['runner_base',
        'runner_iter'], help=
        """Runner to use. The "runner_base" uses epoch-based training while iter-based
            runner runs based on iters. Default: runner_base"""
        )
    validator.add_argument('train_dataset_ratios', type=Dict[str, float],
        help=
        """Ratios of training dataset. This is used in iteration-based runner.
        Do not support for epoch-based runner because how to define an epoch becomes tricky.
        Default: None"""
        )
    validator.add_argument('max_iters', type=float, help=
        'Maximum number of iterations to run.')
    validator.add_argument('max_epoch', type=int, help=
        'Maximum number of epochs to run.')
    validator.add_argument('iters_per_inner_epoch', type=float, help=
        'Number of iterations per inner epoch. This is required when runner is runner_iter.'
        )
    lr_scheds_choices = registry.list_lr_schedulers()
    validator.add_argument('lr_sched', type=str, choices=lr_scheds_choices,
        help='Learning rate scheduler to use, from {}'.format(
        lr_scheds_choices))
    task_choices = registry.list_tasks()
    validator.add_argument('task', type=str, choices=task_choices, help=
        'Task to use, from {}'.format(task_choices))
    validator.add_argument('init_lr', type=float, help=
        'Initial learning rate. This will be the learning rate after warmup and before decay.'
        )
    validator.add_argument('min_lr', type=float, help=
        'Minimum learning rate (after decay).')
    validator.add_argument('warmup_lr', type=float, help=
        'Starting learning rate for warmup.')
    validator.add_argument('lr_decay_rate', type=float, help=
        'Learning rate decay rate. Required if using a decaying learning rate scheduler.'
        )
    validator.add_argument('weight_decay', type=float, help=
        'Weight decay rate.')
    validator.add_argument('batch_size_train', type=int, help=
        'Training batch size.')
    validator.add_argument('batch_size_eval', type=int, help=
        'Evaluation batch size, including validation and testing.')
    validator.add_argument('num_workers', help=
        'Number of workers for data loading.')
    validator.add_argument('warmup_steps', type=int, help=
        'Number of warmup steps. Required if a warmup schedule is used.')
    validator.add_argument('seed', type=int, help='Random seed.')
    validator.add_argument('output_dir', type=str, help=
        'Output directory to save checkpoints and logs.')
    validator.add_argument('evaluate', help=
        'Whether to only evaluate the model. If true, training will not be performed.'
        )
    validator.add_argument('train_splits', type=list, help=
        'Splits to use for training.')
    validator.add_argument('valid_splits', type=list, help=
        'Splits to use for validation. If not provided, will skip the validation.'
        )
    validator.add_argument('test_splits', type=list, help=
        'Splits to use for testing. If not provided, will skip the testing.')
    validator.add_argument('accum_grad_iters', type=int, help=
        'Number of iterations to accumulate gradient for.')
    validator.add_argument('device', type=str, choices=['cpu', 'cuda'],
        help="Device to use. Support 'cuda' or 'cpu' as for now.")
    validator.add_argument('world_size', type=int, help=
        'Number of processes participating in the job.')
    validator.add_argument('dist_url', type=str)
    validator.add_argument('distributed', type=bool)
    validator.add_argument('use_dist_eval_sampler', type=bool, help=
        'Whether to use distributed sampler during evaluation or not.')
    validator.add_argument('max_len', type=int, help=
        'Maximal length of text output.')
    validator.add_argument('min_len', type=int, help=
        'Minimal length of text output.')
    validator.add_argument('num_beams', type=int, help=
        'Number of beams used for beam search.')
    validator.add_argument('num_ans_candidates', type=int, help=
        'For ALBEF and BLIP, these models first rank answers according to likelihood to select answer candidates.'
        )
    validator.add_argument('inference_method', type=str, choices=[
        'genearte', 'rank'], help=
        'Inference method to use for question answering. If rank, requires a answer list.'
        )
    validator.add_argument('k_test', type=int, help=
        'Number of top k most similar samples from ITC/VTC selection to be tested.'
        )
    return validator
