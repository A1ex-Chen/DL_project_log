def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.

    Returns: ServeCommand
    """
    return ConvertCommand(args.model_type, args.tf_checkpoint, args.
        pytorch_dump_output, args.config, args.finetuning_task_name)
