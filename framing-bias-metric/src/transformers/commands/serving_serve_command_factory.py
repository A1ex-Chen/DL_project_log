def serve_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.

    Returns: ServeCommand
    """
    nlp = pipeline(task=args.task, model=args.model if args.model else None,
        config=args.config, tokenizer=args.tokenizer, device=args.device)
    return ServeCommand(nlp, args.host, args.port, args.workers)
