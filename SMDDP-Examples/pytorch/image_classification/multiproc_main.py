def main():
    args = parse_args()
    dist_world_size = args.nproc_per_node * args.nnodes
    current_env = os.environ.copy()
    current_env['MASTER_ADDR'] = args.master_addr
    current_env['MASTER_PORT'] = str(args.master_port)
    current_env['WORLD_SIZE'] = str(dist_world_size)
    processes = []
    for local_rank in range(0, args.nproc_per_node):
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env['RANK'] = str(dist_rank)
        current_env['LOCAL_RANK'] = str(local_rank)
        cmd = [sys.executable, '-u', args.training_script
            ] + args.training_script_args
        print(cmd)
        stdout = None if local_rank == 0 else open('GPU_' + str(local_rank) +
            '.log', 'w')
        process = subprocess.Popen(cmd, env=current_env, stdout=stdout,
            stderr=stdout)
        processes.append(process)
    try:
        up = True
        error = False
        while up and not error:
            up = False
            for p in processes:
                ret = p.poll()
                if ret is None:
                    up = True
                elif ret != 0:
                    error = True
            time.sleep(1)
        if error:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            exit(1)
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        raise
    except SystemExit:
        for p in processes:
            p.terminate()
        raise
    except:
        for p in processes:
            p.terminate()
        raise
