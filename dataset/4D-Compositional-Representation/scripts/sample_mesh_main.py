def main(args):
    hids = os.listdir(os.path.join(args.in_folder))
    seq_folders = []
    for hid in hids:
        seq_folders.extend(glob.glob(os.path.join(args.in_folder, hid, '*')))
    seq_folders.sort()
    print('Total number of sequences: ', len(seq_folders))
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), seq_folders)
    else:
        for p in seq_folders:
            process_path(p, args)
