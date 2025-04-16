def parse_args():
    parser = argparse.ArgumentParser(description=
        'Inference for text to audio generation task.')
    parser.add_argument('--original_args', type=str, default=None, help=
        'Path for summary jsonl file saved during training.')
    parser.add_argument('--model', type=str, default=None, help=
        'Path for saved model bin file.')
    parser.add_argument('--num_teacher_steps', type=int, default=80, help=
        'How many teacher denoising steps for generation.')
    parser.add_argument('--cfg_weight', type=int, default=4.0, help=
        'Classifier-free guidance weight for the student model.')
    parser.add_argument('--use_ema', action='store_true', default=False,
        help='Use the EMA model for inference.')
    parser.add_argument('--use_edm', action='store_true', default=False,
        help="Use EDM's solver and scheduler.")
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    args = parser.parse_args()
    return args
