def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description=
        'PyTorch distributed training launch helper utilty that will spawn up multiple distributed processes'
        )
    parser.add_argument('--nnodes', type=int, default=1, help=
        'The number of nodes to use for distributed training')
    parser.add_argument('--node_rank', type=int, default=0, help=
        'The rank of the node for multi-node distributed training')
    parser.add_argument('--nproc_per_node', type=int, default=1, help=
        'The number of processes to launch on each node, for GPU training, this is recommended to be set to the number of GPUs in your system so that each process can be bound to a single GPU.'
        )
    parser.add_argument('--master_addr', default='127.0.0.1', type=str,
        help=
        "Master node (rank 0)'s address, should be either the IP address or the hostname of node 0, for single node multi-proc training, the --master_addr can simply be 127.0.0.1"
        )
    parser.add_argument('--master_port', default=29500, type=int, help=
        "Master node (rank 0)'s free port that needs to be used for communciation during distributed training"
        )
    parser.add_argument('training_script', type=str, help=
        'The full path to the single GPU training program/script to be launched in parallel, followed by all the arguments for the training script'
        )
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()
