def main():
    args = parse_args()
    NSOCKETS = args.nsockets_per_node
    NGPUS_PER_SOCKET = args.nproc_per_node // args.nsockets_per_node
    NCORES_PER_GPU = args.ncores_per_socket // NGPUS_PER_SOCKET
    dist_world_size = args.nproc_per_node * args.nnodes
    current_env = os.environ.copy()
    current_env['MASTER_ADDR'] = args.master_addr
    current_env['MASTER_PORT'] = str(args.master_port)
    current_env['WORLD_SIZE'] = str(dist_world_size)
    processes = []
    for local_rank in range(0, args.nproc_per_node):
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env['RANK'] = str(dist_rank)
        cpu_ranges = [local_rank * NCORES_PER_GPU, (local_rank + 1) *
            NCORES_PER_GPU - 1, local_rank * NCORES_PER_GPU + 
            NCORES_PER_GPU * NGPUS_PER_SOCKET * NSOCKETS, (local_rank + 1) *
            NCORES_PER_GPU + NCORES_PER_GPU * NGPUS_PER_SOCKET * NSOCKETS - 1]
        numactlargs = []
        if args.no_hyperthreads:
            numactlargs += ['--physcpubind={}-{}'.format(*cpu_ranges[0:2])]
        else:
            numactlargs += ['--physcpubind={}-{},{}-{}'.format(*cpu_ranges)]
        if not args.no_membind:
            memnode = local_rank // NGPUS_PER_SOCKET
            numactlargs += ['--membind={}'.format(memnode)]
        cmd = ['/usr/bin/numactl'] + numactlargs + [sys.executable, '-u',
            args.training_script, '--local_rank={}'.format(local_rank)
            ] + args.training_script_args
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)
    for process in processes:
        process.wait()
