def add_new_model_command_factory(args: Namespace):
    return AddNewModelCommand(args.testing, args.testing_file, path=args.path)
