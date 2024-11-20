@staticmethod
def add_model_specific_args(parser, root_dir):
    BaseTransformer.add_model_specific_args(parser, root_dir)
    add_generic_args(parser, root_dir)
    parser.add_argument('--max_source_length', default=1024, type=int, help
        =
        'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.'
        )
    parser.add_argument('--max_target_length', default=56, type=int, help=
        'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.'
        )
    parser.add_argument('--val_max_target_length', default=142, type=int,
        help=
        'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.'
        )
    parser.add_argument('--test_max_target_length', default=142, type=int,
        help=
        'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.'
        )
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_embeds', action='store_true')
    parser.add_argument('--sortish_sampler', action='store_true', default=False
        )
    parser.add_argument('--overwrite_output_dir', action='store_true',
        default=False)
    parser.add_argument('--max_tokens_per_batch', type=int, default=None)
    parser.add_argument('--logger_name', type=str, choices=['default',
        'wandb', 'wandb_shared'], default='default')
    parser.add_argument('--n_train', type=int, default=-1, required=False,
        help='# examples. -1 means use all.')
    parser.add_argument('--n_val', type=int, default=-1, required=False,
        help='# examples. -1 means use all.')
    parser.add_argument('--n_test', type=int, default=-1, required=False,
        help='# examples. -1 means use all.')
    parser.add_argument('--task', type=str, default='summarization',
        required=False, help='# examples. -1 means use all.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
        required=False)
    parser.add_argument('--task_loss_ratio', type=float, default=0.0,
        required=False)
    parser.add_argument('--src_lang', type=str, default='', required=False)
    parser.add_argument('--tgt_lang', type=str, default='', required=False)
    parser.add_argument('--eval_beams', type=int, default=None, required=False)
    parser.add_argument('--val_metric', type=str, default=None, required=
        False, choices=['bleu', 'rouge2', 'loss', None])
    parser.add_argument('--eval_max_gen_length', type=int, default=None,
        help='never generate more than n tokens')
    parser.add_argument('--save_top_k', type=int, default=1, required=False,
        help='How many checkpoints to save')
    parser.add_argument('--early_stopping_patience', type=int, default=-1,
        required=False, help=
        '-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.'
        )
    parser.add_argument('--custom_pred_file_suffix', type=str, default='',
        required=False, help='Custom prefix for the prediction.txt file')
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--top_p', type=float, default=1.0, required=False,
        help=
        'If set to float < 1, only the most probable tokens with                             probabilities that add up to :obj:`top_p` or higher are kept for generation.'
        )
    parser.add_argument('--top_k', type=int, default=50, required=False,
        help=
        'The number of highest probability vocabulary tokens to keep for top-k-filtering.'
        )
    parser.add_argument('--length_penalty', type=float, default=1.0,
        required=False, help=
        'Exponential penalty to the length. 1.0 means no penalty.                                     Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to                                     encourage the model to produce longer sequences.'
        )
    parser.add_argument('--temperature', type=float, default=1.0, required=
        False, help='The value used to module the next token probabilities')
    parser.add_argument('--extra_task', type=str, default='', required=
        False, help='Name of additional extra task for MT setting')
    parser.add_argument('--num_return_sequences', type=int, default=1, help
        ='# of generation seq to return for each sample')
    return parser
