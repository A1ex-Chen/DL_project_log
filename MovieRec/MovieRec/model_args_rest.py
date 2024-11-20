import argparse

parser = argparse.ArgumentParser(description='model arg for bert')

# device #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# Model
parser.add_argument('--model_init_seed', type=int, default=0)
parser.add_argument('--bert_max_len', type=int, default=100, help='Length of sequence for bert')
parser.add_argument('--bert_hidden_units', type=int, default=256, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=4, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')


args = parser.parse_args()