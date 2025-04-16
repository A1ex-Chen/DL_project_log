def parse_args():
    parser = argparse.ArgumentParser(description=
        'Inference for text to audio generation task.')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=2,
        help=
        'Specifies the stage of the disillation. Must be 1 or 2. Stage 2 corresponds to consistency distillation'
        )
    parser.add_argument('--seed', type=int, default=0, help=
        'Random seed for reproducibility.')
    parser.add_argument('--use_bf16', action='store_true', default=False,
        help='Use bf16 for the LDM model.')
    parser.add_argument('--original_args', type=str, default=None, help=
        'Path for summary jsonl file saved during training.')
    parser.add_argument('--model', type=str, default=None, help=
        'Path for saved model bin file.')
    parser.add_argument('--test_file', type=str, default=
        'data/test_audiocaps_subset.json', help=
        'json file containing the test prompts for generation.')
    parser.add_argument('--text_key', type=str, default='captions', help=
        'Key containing the text in the json file.')
    parser.add_argument('--test_references', type=str, default=
        'data/audiocaps_test_references/subset', help=
        'Folder containing the test reference wav files.')
    parser.add_argument('--num_steps', type=int, default=200, help=
        'How many denoising steps for generation.')
    parser.add_argument('--use_ema', action='store_true', default=False,
        help='Use the EMA model for inference.')
    parser.add_argument('--use_edm', action='store_true', default=False,
        help="Use EDM's solver and scheduler.")
    parser.add_argument('--use_karras', action='store_true', default=False,
        help='Use Karras noise schedule. Only effective when use_edm is True.')
    parser.add_argument('--guidance_scale_input', type=float, default=3,
        help='Classifier-free guidance scale to be fed into the U-Net.')
    parser.add_argument('--guidance_scale_post', type=float, default=1,
        help=
        'Scale of classifier-free guidance to be performed on the model generation.'
        )
    parser.add_argument('--batch_size', type=int, default=8, help=
        'Batch size for generation.')
    parser.add_argument('--num_test_instances', type=int, default=-1, help=
        'How many test instances to evaluate.')
    parser.add_argument('--query_teacher', action='store_true', default=
        False, help=
        'If True, calculate the MSE loss of the generation w.r.t. the teacher.'
        )
    return parser.parse_args()
