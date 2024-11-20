def run_command_factory(args):
    nlp = pipeline(task=args.task, model=args.model if args.model else None,
        config=args.config, tokenizer=args.tokenizer, device=args.device)
    format = try_infer_format_from_ext(args.input
        ) if args.format == 'infer' else args.format
    reader = PipelineDataFormat.from_str(format=format, output_path=args.
        output, input_path=args.input, column=args.column if args.column else
        nlp.default_input_names, overwrite=args.overwrite)
    return RunCommand(nlp, reader)
