def add_generic_args(parser, root_dir) ->None:
    parser.add_argument('--output_dir', default=None, type=str, required=
        True, help=
        'The output directory where the model predictions and checkpoints will be written.'
        )
    parser.add_argument('--fp16', action='store_true', help=
        'Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit'
        )
    parser.add_argument('--fp16_opt_level', type=str, default='O2', help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at https://nvidia.github.io/apex/amp.html"
        )
    parser.add_argument('--n_tpu_cores', dest='tpu_cores', type=int)
    parser.add_argument('--max_grad_norm', dest='gradient_clip_val',
        default=1.0, type=float, help='Max gradient norm')
    parser.add_argument('--do_train', action='store_true', help=
        'Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help=
        'Whether to run predictions on the test set.')
    parser.add_argument('--gradient_accumulation_steps', dest=
        'accumulate_grad_batches', type=int, default=1, help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
        )
    parser.add_argument('--seed', type=int, default=42, help=
        'random seed for initialization')
    parser.add_argument('--data_dir', default=None, type=str, required=True,
        help=
        'The input data dir. Should contain the training files for the CoNLL-2003 NER task.'
        )
