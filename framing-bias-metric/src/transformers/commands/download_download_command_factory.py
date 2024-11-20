def download_command_factory(args):
    return DownloadCommand(args.model, args.cache_dir, args.force)
