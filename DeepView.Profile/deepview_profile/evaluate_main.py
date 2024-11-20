def main():
    parser = argparse.ArgumentParser(prog='deepview-evaluate', description=
        'DeepView Evaluation Tool')
    subparsers = parser.add_subparsers(title='Commands')
    deepview_profile.commands.measurements.register_command(subparsers)
    deepview_profile.commands.prediction_models.register_command(subparsers)
    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help()
        sys.exit(1)
    args.func(args)
