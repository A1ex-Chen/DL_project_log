def update_argparser(parser):
    parser.add_argument('--config', default='resnet50', type=str, required=
        True, help='Network to deploy')
    parser.add_argument('--checkpoint', default=None, type=str, help=
        'The checkpoint of the model. ')
    parser.add_argument('--classes', type=int, default=1000, help=
        'Number of classes')
    parser.add_argument('--precision', type=str, default='fp32', choices=[
        'fp32', 'fp16'], help='Inference precision')
